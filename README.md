# [PYTORCH] PPO & GRPO for playing Flappy Bird

Train an agent to play Flappy Bird from raw pixels with two policy-gradient methods:

- **PPO** — Actor-Critic: shared CNN backbone, policy head + value head, advantage from GAE.
- **GRPO** — Actor only: no critic, advantage from group statistics `A_i = (R_i − mean) / std` over a group of episodes.

Both use the same CNN backbone as the original DQN (input: 4 stacked 84×84 binary frames, output: 2 actions `[idle, flap]`).
The step-by-step walkthrough is in [`learn.ipynb`](learn.ipynb).

## Demo

<p align="center">
  <img src="demo/ppo.gif" width=280>&nbsp;&nbsp;&nbsp;&nbsp;<img src="demo/grpo.gif" width=280><br/>
  <i>Left: PPO &nbsp;·&nbsp; Right: GRPO</i>
</p>

> The gifs are produced by `test.py --record` from trained checkpoints (see [Test](#test)).

## Project structure

```
configs/default.yaml   hyper-parameters for PPO and GRPO
src/flappy_bird.py     game environment (pygame)
src/utils.py           frame pre-processing (resize 84x84, grayscale, binary)
src/modern_rl.py       ActorCriticNetwork (PPO) and ActorOnlyNetwork (GRPO)
src/rollout.py         build 4-frame state, play / evaluate episodes
train.py               train PPO or GRPO, log to Weights & Biases
test.py                load a checkpoint, watch it play, evaluate, record gif
learn.ipynb            walkthrough notebook
```

## Requirements

```
pip install torch numpy opencv-python pygame-ce pyyaml pillow wandb
```

## Train

```
python train.py --algo ppo  --config configs/default.yaml
python train.py --algo grpo --config configs/default.yaml
```

Any config key can be overridden with `--set`:

```
python train.py --algo grpo --set train.max_iters=50 device=cuda:1 wandb.mode=offline
```

- Training runs headless (no game window) and without the 30 FPS limit of the game.
- Checkpoints are saved to `results/{algo}-{timestamp}/`: `{algo}_best.pt` (best eval score), `{algo}_iter{N}.pt` (every `train.save_every` iterations), `{algo}_last.pt`.
- Total training time (minutes) is printed at the end.

### Training on Kaggle (2 GPUs)

One notebook run trains PPO on `cuda:0` and GRPO on `cuda:1` in parallel. Add your W&B API key as a Kaggle secret named `WANDB_API_KEY`, then run:

```python
import os, subprocess, time
from kaggle_secrets import UserSecretsClient

os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
subprocess.run("pip install -q pygame-ce wandb", shell=True)

MAX_ITERS = 50
ALGOS = ["ppo", "grpo"]                 # ppo -> cuda:0, grpo -> cuda:1

start = time.time()
cmd = "python train.py --algo {} --config configs/default.yaml --set train.max_iters={} device=cuda:{}"
procs = [subprocess.Popen(cmd.format(a, MAX_ITERS, i), shell=True,
                          stdout=open(f"{a}.out", "w"), stderr=subprocess.STDOUT)
         for i, a in enumerate(ALGOS)]
for p in procs:
    p.wait()
print(f"train: {(time.time() - start) / 60:.1f} min")
```

## Test

Watch the agent play (opens a game window):

```
python test.py --checkpoint results/ppo-xxx/ppo_best.pt
```

Final evaluation (no window, fast):

```
python test.py --checkpoint results/ppo-xxx/ppo_best.pt --episodes 50 --no-render
```

Record the demo gif:

```
python test.py --checkpoint results/ppo-xxx/ppo_best.pt  --episodes 1 --record demo/ppo.gif
python test.py --checkpoint results/grpo-xxx/grpo_best.pt --episodes 1 --record demo/grpo.gif
```

| Option | Default | Meaning |
|---|---|---|
| `--episodes` | 5 | number of episodes |
| `--max-steps` | 10000 | frame cap per episode |
| `--fps` | 30 | game speed when rendering |
| `--no-render` | off | no window, no FPS limit |
| `--sample` | off | sample actions instead of argmax |
| `--record PATH` | – | save the first episode as a gif |

`test.py` prints score / length / return of every episode, then a summary table (mean, std, median, max) and `cap_rate`.

## Metrics (Weights & Biases)

Project `FlappyBird-RL`, one run per algorithm (grouped by `algo`). X-axis: `global_step` = number of game frames used for training.

| Metric | Meaning |
|---|---|
| `train/score`, `train/length`, `train/return` | per training episode (sampled actions): pipes passed, frames survived, total reward |
| `train/entropy` | policy entropy |
| `eval/score_mean`, `eval/length_mean`, `eval/return_mean` | greedy (argmax) evaluation, averaged over `eval.episodes` episodes |
| `eval/cap_rate` | fraction of eval episodes reaching `eval.max_steps` without dying |

Also logged: `train/policy_loss`, `train/approx_kl`, `train/clip_frac`, `train/grad_norm`, `train/value_loss` (PPO), `train/group_reward_std` (GRPO).

## Reward

| Event | Reward |
|---|---|
| each frame alive | +0.1 |
| passing a pipe | +1 |
| dying | −1 |

## Original DQN version

The original Deep Q-learning code is kept as `old_train.py`, `old_test.py`, `OLD_README.md`.
