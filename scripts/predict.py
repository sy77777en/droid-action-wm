#!/usr/bin/env python3
"""Predict 16 future frames (5 Hz) from one exterior-camera image and a 16x7 action chunk.

  python scripts/predict.py --obs obs.png --actions actions_raw.npy --out out/
  # or from absolute EEF poses: --poses poses.npy (17x6) --gripper gripper.npy (17,)
  # longer horizon: actions (16*K, 7) -> autoregressive rollout
"""
import argparse, os, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from droid_wm import DroidWorldModel, actions_from_poses  # noqa: E402


def save(frames, out, fps=5):
    os.makedirs(out, exist_ok=True)
    for i, im in enumerate(frames):
        im.save(os.path.join(out, f"frame_{i + 1:02d}.png"))
    try:
        import imageio
        imageio.mimsave(os.path.join(out, "pred.mp4"), [np.asarray(f) for f in frames], fps=fps)
    except Exception as e:  # noqa: BLE001
        print("[warn] mp4 not written:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--actions", help="raw action rows (16*K, 7) .npy")
    ap.add_argument("--poses", help="absolute EEF poses (17, 6) .npy, frames 0..16 at 5 Hz")
    ap.add_argument("--gripper", help="gripper position (17,) .npy")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.actions:
        act = np.load(a.actions)
    else:
        act = actions_from_poses(np.load(a.poses), np.load(a.gripper))
    w = a.weights
    wm = DroidWorldModel.from_pretrained(
        ckpt=f"{w}/droid-action-wm/droid_wm_fullft_s64730.safetensors",
        wan_root=f"{w}/Wan2.2-TI2V-5B",
        stats_json=f"{w}/droid-action-wm/action_stats_full_interp.json",
        num_inference_steps=a.steps)
    obs = Image.open(a.obs)
    frames = wm.predict(obs, act, seed=a.seed) if len(act) == 16 else wm.rollout(obs, act, seed=a.seed)
    save(frames, a.out)
    print(f"{len(frames)} frames -> {a.out}")


if __name__ == "__main__":
    main()
