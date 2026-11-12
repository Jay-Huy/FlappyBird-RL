"""
Load a trained PPO / GRPO checkpoint and play Flappy Bird.

Watch it play (opens a game window):
  python test.py --checkpoint results/ppo-xxx/ppo_best.pt
Final evaluation (no window, no FPS limit):
  python test.py --checkpoint results/ppo-xxx/ppo_best.pt --episodes 50 --no-render
Record a gif for the README:
  python test.py --checkpoint results/ppo-xxx/ppo_best.pt --episodes 1 --record demo/ppo.gif
"""
import argparse
import os
import sys


def get_args():
    parser = argparse.ArgumentParser("Play Flappy Bird with a trained PPO / GRPO agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt file saved by train.py")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10000, help="Cap per episode (counts toward cap_rate)")
    parser.add_argument("--fps", type=int, default=30, help="Game speed when rendering")
    parser.add_argument("--no-render", action="store_true", help="No window, run as fast as possible")
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of argmax")
    parser.add_argument("--record", type=str, default=None, help="Save the first episode as a gif, e.g. demo/ppo.gif")
    parser.add_argument("--record-max-frames", type=int, default=600, help="Max game frames written to the gif")
    parser.add_argument("--device", type=str, default=None, help="Default: cuda if available else cpu")
    return parser.parse_args()


args = get_args()
if args.no_render:
    # Must be set before importing FlappyBird (it opens the window at import time)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import torch
from PIL import Image

from src.flappy_bird import FlappyBird
from src.modern_rl import ActorCriticNetwork, ActorOnlyNetwork
from src.rollout import disable_frame_limit, play_episode, summarize


class GifRecorder:
    """Keep every 2nd frame, half resolution, gif at 15 fps (= real-time for a 30 fps game)."""
    def __init__(self, max_frames):
        self.frames, self.count, self.max_frames = [], 0, max_frames

    def __call__(self, raw_image):
        if self.count < self.max_frames and self.count % 2 == 0:
            img = Image.fromarray(raw_image.transpose(1, 0, 2))  # array3d is (W, H, 3) -> (H, W, 3)
            self.frames.append(img.resize((img.width // 2, img.height // 2)))
        self.count += 1

    def save(self, path):
        if not self.frames:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.frames[0].save(path, save_all=True, append_images=self.frames[1:], duration=1000 // 15, loop=0)
        print(f"Saved gif: {path} ({len(self.frames)} frames)")


def main():
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    algo = ckpt["algo"]
    model = (ActorCriticNetwork() if algo == "ppo" else ActorOnlyNetwork()).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded {algo.upper()} checkpoint: {args.checkpoint} "
          f"(iteration {ckpt.get('iteration')}, {ckpt.get('global_step')} training frames)")

    if args.no_render:
        disable_frame_limit()
    else:
        FlappyBird.fps = args.fps

    closed = {"flag": False}

    def should_stop():
        if args.no_render:
            return False
        for event in pygame.event.get(pygame.QUIT):
            closed["flag"] = True
        return closed["flag"]

    recorder = GifRecorder(args.record_max_frames) if args.record else None
    results = []
    for ep in range(1, args.episodes + 1):
        on_frame = recorder if (recorder is not None and ep == 1) else None
        r = play_episode(model, device, greedy=not args.sample, max_steps=args.max_steps,
                         on_frame=on_frame, should_stop=should_stop)
        if closed["flag"]:
            print("Window closed.")
            break
        results.append(r)
        print(f"Episode {ep} | score {r['score']} | length {r['length']} | return {r['return']:.1f}"
              + (" | hit max-steps" if r["capped"] else ""))

    if recorder is not None:
        recorder.save(args.record)

    if results:
        s = summarize(results)
        print(f"\nSummary over {len(results)} episodes ({'sample' if args.sample else 'argmax'})")
        print(f"{'':8s}{'mean':>10s}{'std':>10s}{'median':>10s}{'max':>10s}")
        for key in ("score", "length", "return"):
            v = s[key]
            print(f"{key:8s}{v['mean']:10.2f}{v['std']:10.2f}{v['median']:10.2f}{v['max']:10.2f}")
        print(f"cap_rate {s['cap_rate']:.2f}  (episodes reaching --max-steps={args.max_steps} without dying)")

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
