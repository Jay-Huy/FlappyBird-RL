"""
Shared helpers for train.py and test.py.
Same logic as learn.ipynb, gathered into functions:
- build the 4-frame state from game frames
- play one episode and measure score / length / return
"""
import numpy as np
import torch

from src.flappy_bird import FlappyBird
from src.utils import pre_processing

IMAGE_SIZE = 84


class _NoClock:
    def tick(self, *args, **kwargs):
        return 0


def disable_frame_limit():
    """FlappyBird.next_frame calls fps_clock.tick(30), i.e. waits ~1/30 s per frame.
    Replace the clock so the game runs as fast as the CPU allows (training / headless eval)."""
    FlappyBird.fps_clock = _NoClock()


def frame_to_tensor(raw_image, env, device):
    """Crop the floor, pre_processing to 84x84 binary -> tensor [1, 84, 84]."""
    cropped = raw_image[:env.screen_width, :int(env.base_y)]
    processed = pre_processing(cropped, IMAGE_SIZE, IMAGE_SIZE)
    return torch.from_numpy(processed).to(device)


def initial_state(env, device):
    """Step once with action 0 (idle), stack 4 identical frames -> [1, 4, 84, 84]."""
    raw_image, _, _ = env.next_frame(0)
    frame = frame_to_tensor(raw_image, env, device)
    return torch.cat(tuple(frame for _ in range(4)))[None, :, :, :]


def next_state(state, raw_image, env, device):
    """Sliding window: drop the oldest frame, append the newest -> [1, 4, 84, 84]."""
    frame = frame_to_tensor(raw_image, env, device)
    return torch.cat((state[0, 1:, :, :], frame))[None, :, :, :]


def policy_logits(model, state):
    """ActorCriticNetwork returns (logits, value); ActorOnlyNetwork returns logits."""
    out = model(state)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def play_episode(model, device, greedy=True, max_steps=10000, on_frame=None, should_stop=None):
    """
    Play 1 episode on a fresh FlappyBird until game over or max_steps.
    greedy=True -> argmax action, else sample from the policy.
    on_frame(raw_image): optional callback (e.g. record gif).
    should_stop(): optional callback, return True to abort (e.g. window closed).
    Returns dict: score (pipes passed), length (frames), return (sum reward), capped (hit max_steps).
    """
    env = FlappyBird()
    state = initial_state(env, device)
    score, length, total_reward, terminal = 0, 0, 0.0, False

    while length < max_steps:
        logits = policy_logits(model, state)
        if greedy:
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            action = int(torch.distributions.Categorical(logits=logits).sample().item())

        raw_image, reward, terminal = env.next_frame(action)
        if on_frame is not None:
            on_frame(raw_image)

        length += 1
        total_reward += reward
        if reward == 1:
            score += 1
        if terminal or (should_stop is not None and should_stop()):
            break
        state = next_state(state, raw_image, env, device)

    return {"score": score, "length": length, "return": total_reward,
            "capped": (not terminal) and length >= max_steps}


def evaluate(model, device, episodes=5, max_steps=10000, greedy=True):
    """Play several episodes, return the list of per-episode dicts."""
    was_training = model.training
    model.eval()
    results = [play_episode(model, device, greedy=greedy, max_steps=max_steps) for _ in range(episodes)]
    if was_training:
        model.train()
    return results


def summarize(results):
    """mean / std / median / max for score, length, return + cap_rate."""
    out = {}
    for key in ("score", "length", "return"):
        values = np.array([r[key] for r in results], dtype=np.float64)
        out[key] = {"mean": values.mean(), "std": values.std(), "median": float(np.median(values)), "max": values.max()}
    out["cap_rate"] = float(np.mean([r["capped"] for r in results]))
    return out
