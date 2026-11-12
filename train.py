"""
Train PPO or GRPO on Flappy Bird, log to Weights & Biases.
Training logic is taken from learn.ipynb (section 7).

Usage:
  python train.py --algo ppo  --config configs/default.yaml --set train.max_iters=50 device=cuda:0
  python train.py --algo grpo --config configs/default.yaml --set train.max_iters=50 device=cuda:1
"""
import os
# Headless: FlappyBird opens a window at import time -> use SDL dummy drivers (must be set before import)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.flappy_bird import FlappyBird
from src.modern_rl import ActorCriticNetwork, ActorOnlyNetwork
from src.rollout import (disable_frame_limit, initial_state, next_state, evaluate, summarize)


# ==============================================================================
# CONFIG
# ==============================================================================
def load_config(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def get_args():
    parser = argparse.ArgumentParser("Train PPO / GRPO on Flappy Bird")
    parser.add_argument("--algo", type=str, choices=["ppo", "grpo"], required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="Overrides, e.g. train.max_iters=50 device=cuda:1")
    return parser.parse_args()


# ==============================================================================
# PPO (learn.ipynb, 7.1)
# ==============================================================================
def train_ppo_iteration(model, optimizer, game_state, current_state, episode_tracker,
                        num_steps=128, gamma=0.99, gae_lambda=0.95,
                        clip_eps=0.2, ppo_epochs=4, batch_size=32,
                        c_val=0.5, c_ent=0.01, max_grad_norm=0.5):
    """
    One full PPO iteration. episode_tracker carries the unfinished episode across iterations
    (a 128-step rollout can cut an episode in two). Returns (state, metrics, finished_episodes).
    """
    device = next(model.parameters()).device

    # PHASE 1: ROLLOUT COLLECTION
    b_states, b_actions, b_log_probs, b_rewards, b_dones, b_values = [], [], [], [], [], []
    finished_episodes = []
    state = current_state.to(device)

    for step in range(num_steps):
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(state)

        raw_img, reward, terminal = game_state.next_frame(action.item())
        next_st = next_state(state, raw_img, game_state, device)

        b_states.append(state)
        b_actions.append(action)
        b_log_probs.append(log_prob)
        b_rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
        b_dones.append(torch.tensor(1.0 if terminal else 0.0, dtype=torch.float32, device=device))
        b_values.append(value)

        # Episode bookkeeping (for logging only)
        episode_tracker["length"] += 1
        episode_tracker["return"] += reward
        if reward == 1:
            episode_tracker["score"] += 1
        if terminal:
            finished_episodes.append(dict(episode_tracker))
            episode_tracker.update(score=0, length=0, **{"return": 0.0})

        state = next_st

    with torch.no_grad():
        _, _, _, next_val = model.get_action_and_value(state)

    # PHASE 2: GAE
    b_rewards_t = torch.stack(b_rewards).view(-1)
    b_dones_t = torch.stack(b_dones).view(-1)
    b_values_t = torch.stack(b_values).view(-1)
    next_val = next_val.view(-1)

    advantages = torch.zeros(num_steps, device=device)
    last_gae_lam = 0.0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - b_dones_t[t]
            next_value = next_val[0]
        else:
            next_non_terminal = 1.0 - b_dones_t[t]
            next_value = b_values_t[t + 1]
        delta = b_rewards_t[t] + gamma * next_value * next_non_terminal - b_values_t[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

    b_returns = advantages + b_values_t

    flat_states = torch.cat(b_states, dim=0)
    flat_actions = torch.stack(b_actions).view(-1)
    flat_old_log_probs = torch.stack(b_log_probs).view(-1)
    flat_advantages = advantages.view(-1)
    flat_returns = b_returns.view(-1)
    flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

    # PHASE 3 & 4: PPO CLIPPED UPDATE
    num_samples = num_steps
    indices = np.arange(num_samples)
    sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0}
    n_updates = 0

    for epoch in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            mb_idx = indices[start:start + batch_size]

            _, new_log_prob, entropy, new_value = model.get_action_and_value(
                flat_states[mb_idx], flat_actions[mb_idx]
            )
            log_ratio = new_log_prob - flat_old_log_probs[mb_idx]
            ratio = torch.exp(log_ratio)

            mb_adv = flat_advantages[mb_idx]
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * ((new_value.view(-1) - flat_returns[mb_idx].view(-1)) ** 2).mean()
            entropy_loss = -entropy.mean()
            loss = policy_loss + c_val * value_loss + c_ent * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                sums["policy_loss"] += policy_loss.item()
                sums["value_loss"] += value_loss.item()
                sums["entropy"] += entropy.mean().item()
                sums["approx_kl"] += ((ratio - 1) - log_ratio).mean().item()
                sums["clip_frac"] += ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                sums["grad_norm"] += float(grad_norm)
            n_updates += 1

    metrics = {k: v / max(1, n_updates) for k, v in sums.items()}
    metrics["frames"] = num_steps
    return state, metrics, finished_episodes


# ==============================================================================
# GRPO (learn.ipynb, 7.2)
# ==============================================================================
def run_single_episode(model, max_steps=128):
    """Plays 1 episode (sampling) until game over or max_steps. Returns trajectory + episode stats."""
    device = next(model.parameters()).device
    env = FlappyBird()
    state = initial_state(env, device)

    ep_states, ep_actions, ep_log_probs = [], [], []
    total_reward, score = 0.0, 0

    for _ in range(max_steps):
        with torch.no_grad():
            action, log_prob, _ = model.get_action(state)

        next_raw, reward, terminal = env.next_frame(action.item())
        next_st = next_state(state, next_raw, env, device)

        ep_states.append(state)
        ep_actions.append(action)
        ep_log_probs.append(log_prob)
        total_reward += reward
        if reward == 1:
            score += 1

        state = next_st
        if terminal:
            break

    return (torch.cat(ep_states, dim=0),
            torch.stack(ep_actions).view(-1),
            torch.stack(ep_log_probs).view(-1),
            total_reward,
            {"score": score, "length": len(ep_actions), "return": total_reward})


def train_grpo_iteration(model, optimizer, group_size=4, max_steps=128, clip_eps=0.2,
                         grpo_epochs=4, batch_size=32, c_ent=0.01, max_grad_norm=0.5):
    """One full GRPO iteration. Returns (metrics, finished_episodes)."""
    device = next(model.parameters()).device
    model.train()

    # PHASE 1: GROUP ROLLOUT
    group_trajectories, group_rewards, finished_episodes = [], [], []
    for _ in range(group_size):
        states, actions, log_probs, ep_reward, ep_stats = run_single_episode(model, max_steps=max_steps)
        group_trajectories.append({"states": states, "actions": actions,
                                   "old_log_probs": log_probs, "reward": ep_reward})
        group_rewards.append(ep_reward)
        finished_episodes.append(ep_stats)

    # PHASE 2: GROUP ADVANTAGE  A_i = (R_i - mu) / sigma
    r_tensor = torch.tensor(group_rewards, dtype=torch.float32, device=device)
    mean_r = r_tensor.mean()
    std_r = r_tensor.std() + 1e-8
    group_advantages = (r_tensor - mean_r) / std_r

    # PHASE 3: FLATTEN
    all_states, all_actions, all_old_log_probs, all_advantages = [], [], [], []
    for i, traj in enumerate(group_trajectories):
        T = traj["actions"].shape[0]
        all_states.append(traj["states"])
        all_actions.append(traj["actions"])
        all_old_log_probs.append(traj["old_log_probs"])
        all_advantages.append(torch.full((T,), group_advantages[i].item(), device=device))

    flat_states = torch.cat(all_states, dim=0)
    flat_actions = torch.cat(all_actions, dim=0)
    flat_old_log_probs = torch.cat(all_old_log_probs, dim=0)
    flat_advantages = torch.cat(all_advantages, dim=0)

    # PHASE 4: GRPO CLIPPED UPDATE (no critic)
    num_samples = flat_states.shape[0]
    indices = np.arange(num_samples)
    sums = {"policy_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0}
    n_updates = 0

    for epoch in range(grpo_epochs):
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            mb_idx = indices[start:start + batch_size]

            _, new_log_prob, entropy = model.get_action(flat_states[mb_idx], flat_actions[mb_idx])
            log_ratio = new_log_prob - flat_old_log_probs[mb_idx]
            ratio = torch.exp(log_ratio)

            mb_adv = flat_advantages[mb_idx]
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropy.mean()
            loss = policy_loss + c_ent * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                sums["policy_loss"] += policy_loss.item()
                sums["entropy"] += entropy.mean().item()
                sums["approx_kl"] += ((ratio - 1) - log_ratio).mean().item()
                sums["clip_frac"] += ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                sums["grad_norm"] += float(grad_norm)
            n_updates += 1

    metrics = {k: v / max(1, n_updates) for k, v in sums.items()}
    metrics["group_reward_std"] = float(r_tensor.std().item())
    metrics["frames"] = num_samples
    return metrics, finished_episodes


# ==============================================================================
# MAIN LOOP
# ==============================================================================
def save_checkpoint(path, model, algo, iteration, global_step, cfg):
    torch.save({"algo": algo, "state_dict": model.state_dict(), "iteration": iteration,
                "global_step": global_step, "config": cfg}, path)


def main():
    args = get_args()
    cfg = load_config(args.config, args.set)
    algo = args.algo

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() or not str(cfg["device"]).startswith("cuda") else "cpu")
    disable_frame_limit()

    run_name = cfg["wandb"]["run_name"] or f"{algo}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir = os.path.join(cfg["out_dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)

    import wandb
    wandb.init(project=cfg["wandb"]["project"], name=run_name, mode=cfg["wandb"]["mode"],
               config={**cfg, "algo": algo}, group=algo, tags=[algo])
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")

    model = (ActorCriticNetwork() if algo == "ppo" else ActorOnlyNetwork()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    if algo == "ppo":
        env = FlappyBird()
        state = initial_state(env, device)
        episode_tracker = {"score": 0, "length": 0, "return": 0.0}

    max_iters = cfg["train"]["max_iters"]
    global_step = 0
    best_eval = -float("inf")
    start = time.time()
    print(f"[{algo}] device={device} max_iters={max_iters} out_dir={out_dir}", flush=True)

    for it in range(1, max_iters + 1):
        if algo == "ppo":
            p = cfg["ppo"]
            state, metrics, episodes = train_ppo_iteration(
                model, optimizer, env, state, episode_tracker,
                num_steps=p["num_steps"], gamma=p["gamma"], gae_lambda=p["gae_lambda"],
                clip_eps=p["clip_eps"], ppo_epochs=p["epochs"], batch_size=p["batch_size"],
                c_val=p["c_val"], c_ent=p["c_ent"], max_grad_norm=p["max_grad_norm"])
        else:
            g = cfg["grpo"]
            metrics, episodes = train_grpo_iteration(
                model, optimizer, group_size=g["group_size"], max_steps=g["max_steps"],
                clip_eps=g["clip_eps"], grpo_epochs=g["epochs"], batch_size=g["batch_size"],
                c_ent=g["c_ent"], max_grad_norm=g["max_grad_norm"])

        # Training episodes (sampled actions): one log per finished episode
        global_step += metrics.pop("frames")
        for ep in episodes:
            wandb.log({"global_step": global_step, "train/score": ep["score"],
                       "train/length": ep["length"], "train/return": ep["return"]})

        elapsed_min = (time.time() - start) / 60
        wandb.log({"global_step": global_step, "iteration": it, "time/elapsed_min": elapsed_min,
                   **{f"train/{k}": v for k, v in metrics.items()}})

        last = episodes[-1] if episodes else None
        print(f"[{algo}] iter {it:5d}/{max_iters} | frames {global_step:9d} | {elapsed_min:7.1f} min | "
              f"entropy {metrics['entropy']:.4f} | episodes {len(episodes)}"
              + (f" | last score {last['score']} len {last['length']} ret {last['return']:.1f}" if last else ""),
              flush=True)

        # Eval (greedy)
        if it % cfg["train"]["eval_every"] == 0 or it == max_iters:
            results = evaluate(model, device, episodes=cfg["eval"]["episodes"], max_steps=cfg["eval"]["max_steps"])
            s = summarize(results)
            wandb.log({"global_step": global_step,
                       "eval/score_mean": s["score"]["mean"], "eval/length_mean": s["length"]["mean"],
                       "eval/return_mean": s["return"]["mean"], "eval/cap_rate": s["cap_rate"]})
            print(f"[{algo}] EVAL iter {it} | score {s['score']['mean']:.2f} | length {s['length']['mean']:.1f} | "
                  f"return {s['return']['mean']:.2f} | cap_rate {s['cap_rate']:.2f}", flush=True)
            if s["score"]["mean"] > best_eval:
                best_eval = s["score"]["mean"]
                save_checkpoint(os.path.join(out_dir, f"{algo}_best.pt"), model, algo, it, global_step, cfg)

        if it % cfg["train"]["save_every"] == 0:
            save_checkpoint(os.path.join(out_dir, f"{algo}_iter{it}.pt"), model, algo, it, global_step, cfg)

    save_checkpoint(os.path.join(out_dir, f"{algo}_last.pt"), model, algo, max_iters, global_step, cfg)
    total_min = (time.time() - start) / 60
    wandb.summary["train_minutes"] = total_min
    wandb.finish()
    print(f"[{algo}] train: {total_min:.1f} min ({max_iters} iterations, {global_step} frames)", flush=True)


if __name__ == "__main__":
    main()
