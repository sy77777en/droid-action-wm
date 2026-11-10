"""Action and image conventions the checkpoint was trained with.

Clip timing
    DROID records at 15 Hz. The model works at 5 Hz (ACTION_STRIDE = 3 raw steps):
    1 observation frame + 16 predicted frames = 17 frames = 48 raw steps = 3.2 s.

Action row j (j = 0..15), 7 numbers, ROBOT-BASE frame, un-normalised:
    [ x(j+1)-x(j), y(j+1)-y(j), z(j+1)-z(j),          metres
      wrap(rx(j+1)-rx(j)), wrap(ry..), wrap(rz..),    radians, wrapped to (-pi, pi]
      gripper(j+1) ]                                    absolute, 0 = open, 1 = closed
    where pose(j) is DROID's observation/robot_state/cartesian_position
    (xyz + extrinsic XYZ Euler) at 5 Hz frame j, i.e. raw step s + 3j, and gripper(j) is
    observation/robot_state/gripper_position. Row j moves the arm from frame j to
    frame j+1, so it "produces" predicted frame j+1.

    Training used poses interpolated to each frame's camera-capture time (~55 ms
    before the robot_state read); plain step-aligned poses differ by < 1 raw step
    and are fine for policy evaluation.

Normalisation: (row - mean) / std with action_stats_full_interp.json.

Image: one DROID EXTERIOR camera (ext1 or ext2, left image), centre-cropped to 4:3
    and resized to 640x480 with LANCZOS. No wrist camera, no text instruction.
"""
import json

import numpy as np
from PIL import Image

ACTION_STRIDE = 3
N_FUTURE = 16
ACTION_DIM = 7


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def actions_from_poses(cartesian, gripper):
    """cartesian (17, 6) EEF poses at 5 Hz frames 0..16 (frame 0 = observation),
    gripper (17,) in [0, 1]  ->  raw action rows (16, 7)."""
    cart = np.asarray(cartesian, np.float64)
    grip = np.asarray(gripper, np.float64).reshape(-1)
    if cart.shape != (N_FUTURE + 1, 6) or grip.shape != (N_FUTURE + 1,):
        raise ValueError(f"want cartesian (17, 6) and gripper (17,), got {cart.shape} / {grip.shape}")
    d = cart[1:] - cart[:-1]
    d[:, 3:] = wrap(d[:, 3:])
    return np.concatenate([d, grip[1:, None]], 1).astype(np.float32)


class ActionNormalizer:
    def __init__(self, stats_json):
        st = json.load(open(stats_json))
        self.mean = np.asarray(st["mean"], np.float32)
        self.std = np.maximum(np.asarray(st["std"], np.float32), 1e-6)

    def __call__(self, raw):
        raw = np.asarray(raw, np.float32)
        if raw.shape != (N_FUTURE, ACTION_DIM):
            raise ValueError(f"want raw actions (16, 7), got {raw.shape}")
        return (raw - self.mean) / self.std


def preprocess_image(img, width=640, height=480):
    """Centre-crop to 4:3 and resize, matching training. Accepts PIL or HxWx3 uint8."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img, np.uint8))
    img = img.convert("RGB")
    W, H = img.size
    if abs(W * 3 - H * 4) > 1:                 # not already 4:3
        if W * 3 > H * 4:                      # too wide (DROID 1280x720)
            cw = round(H * 4 / 3); x0 = (W - cw) // 2
            img = img.crop((x0, 0, x0 + cw, H))
        else:
            ch = round(W * 3 / 4); y0 = (H - ch) // 2
            img = img.crop((0, y0, W, y0 + ch))
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    return img
