#!/usr/bin/env python3
"""Cut a model-ready clip out of a raw DROID episode (DROID 1.0.1 raw release layout:
<episode>/trajectory.h5, <episode>/metadata_*.json, <episode>/recordings/MP4/<serial>.mp4).

Writes <out>/obs.png, <out>/actions_raw.npy (16, 7), <out>/gt/frame_01..16.png.
Uses exactly the training conventions: frames s, s+3, ..., s+48 of the exterior mp4,
centre-cropped to 4:3, resized to 640x480; actions from poses interpolated to frame
capture time (falls back to step-aligned poses if the timestamps are unusable).

  python scripts/clip_from_droid.py --episode /path/to/episode --cam ext1 --start 60 --out clip/
"""
import argparse, glob, json, os, sys

import h5py
import imageio_ffmpeg
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from droid_wm.actions import ACTION_STRIDE, N_FUTURE, wrap  # noqa: E402


def poses_at_capture(f, serial):
    cart = f["observation/robot_state/cartesian_position"][:].astype(np.float64)
    grip = f["observation/robot_state/gripper_position"][:].astype(np.float64)
    try:
        t_read = f["observation/timestamp/robot_state/read_start"][:].astype(np.float64)
        t_cap = f[f"observation/timestamp/cameras/{serial}_estimated_capture"][:].astype(np.float64)
        T = min(len(cart), len(t_read), len(t_cap))
        if np.isfinite(t_read[:T]).all() and np.isfinite(t_cap[:T]).all() and (np.diff(t_read[:T]) > 0).all():
            ang = np.unwrap(cart[:T, 3:], axis=0)
            p = np.concatenate([cart[:T, :3], ang], 1)
            pc = np.stack([np.interp(t_cap[:T], t_read[:T], p[:, d]) for d in range(6)], 1)
            return pc, np.interp(t_cap[:T], t_read[:T], grip[:T]), "interp"
    except KeyError:
        pass
    return cart, grip, "step"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--cam", default="ext1", choices=["ext1", "ext2"])
    ap.add_argument("--start", type=int, required=True, help="raw 15 Hz step of the observation frame")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    meta = json.load(open(glob.glob(os.path.join(a.episode, "metadata_*.json"))[0]))
    serial = str(meta[f"{a.cam}_cam_serial"])
    mp4 = os.path.join(a.episode, "recordings", "MP4", f"{serial}.mp4")
    with h5py.File(os.path.join(a.episode, "trajectory.h5"), "r") as f:
        pose, grip, mode = poses_at_capture(f, serial)
    idx = a.start + ACTION_STRIDE * np.arange(N_FUTURE + 1)
    if idx[-1] >= len(pose):
        raise SystemExit(f"start {a.start} too late: need step {idx[-1]}, episode has {len(pose)}")
    d = pose[idx[1:]] - pose[idx[:-1]]
    d[:, 3:] = wrap(d[:, 3:])
    act = np.concatenate([d, grip[idx[1:], None]], 1).astype(np.float32)

    g = imageio_ffmpeg.read_frames(mp4, input_params=["-ss", "0"],
                                   output_params=["-vf", r"crop=ih*4/3:ih:(iw-ih*4/3)/2:0"])
    m = next(g); W, H = m["size"]
    want = set(idx.tolist()); frames = {}
    for k, b in enumerate(g):
        if k in want:
            frames[k] = Image.fromarray(np.frombuffer(b, np.uint8).reshape(H, W, 3)).resize((640, 480), Image.LANCZOS)
        if k >= idx[-1]:
            break
    g.close()
    if len(frames) != N_FUTURE + 1:
        raise SystemExit(f"mp4 has too few frames for start {a.start}")
    os.makedirs(os.path.join(a.out, "gt"), exist_ok=True)
    frames[idx[0]].save(os.path.join(a.out, "obs.png"))
    for j, k in enumerate(idx[1:]):
        frames[k].save(os.path.join(a.out, "gt", f"frame_{j + 1:02d}.png"))
    np.save(os.path.join(a.out, "actions_raw.npy"), act)
    json.dump({"episode": a.episode, "cam": a.cam, "start": a.start, "pose_mode": mode,
               "task": meta.get("current_task", "")}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"clip -> {a.out} (pose mode {mode})")


if __name__ == "__main__":
    main()
