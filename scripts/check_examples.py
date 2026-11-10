#!/usr/bin/env python3
"""Sanity check: run the model on the bundled held-out-scene clips and compare PSNR with
the numbers from our paper eval (same seed -> should agree to ~0.01 dB on the same GPU type;
small differences across GPUs / library versions are normal)."""
import argparse, glob, json, os, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from droid_wm import DroidWorldModel  # noqa: E402

def psnr_mean(pred, gt):
    return float(np.mean([10 * np.log10(255.0 ** 2 / np.mean((np.asarray(p, np.float64) - np.asarray(g, np.float64)) ** 2))
                          for p, g in zip(pred, gt)]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--out", default="out/examples")
    a = ap.parse_args()
    w = a.weights
    wm = DroidWorldModel.from_pretrained(
        ckpt=f"{w}/droid-action-wm/droid_wm_fullft_s64730.safetensors",
        wan_root=f"{w}/Wan2.2-TI2V-5B",
        stats_json=f"{w}/droid-action-wm/action_stats_full_interp.json")
    for d in sorted(glob.glob(f"{w}/droid-action-wm/examples/clip*")):
        meta = json.load(open(f"{d}/meta.json"))
        gt = [Image.open(f"{d}/gt/frame_{k:02d}.png").convert("RGB") for k in range(1, 17)]
        pred = wm.predict(Image.open(f"{d}/obs.png"), np.load(f"{d}/actions_raw.npy"),
                          seed=meta["seed_used_in_paper_eval"])
        o = os.path.join(a.out, os.path.basename(d)); os.makedirs(o, exist_ok=True)
        for k, im in enumerate(pred):
            im.save(f"{o}/frame_{k + 1:02d}.png")
        print(f"{os.path.basename(d)}: PSNR {psnr_mean(pred, gt):.2f} dB "
              f"(paper eval: {meta['reference_psnr_frames1_16']:.2f})  task: {meta['source']['task'][:60]}")

if __name__ == "__main__":
    main()
