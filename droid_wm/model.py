"""DroidWorldModel: action-conditioned video world model for DROID (exterior camera).

    wm = DroidWorldModel.from_pretrained(ckpt, wan_root, stats_json)
    frames = wm.predict(obs_image, raw_actions)          # 16 PIL frames, 5 Hz
    frames = wm.rollout(obs_image, raw_actions_long)     # 16*K frames, autoregressive
"""
import contextlib
import io
import types

import numpy as np
import torch

from . import _sampler as base
from .actions import N_FUTURE, ActionNormalizer, preprocess_image


class DroidWorldModel:
    def __init__(self, pipe, normalizer, device, dtype, height=480, width=640,
                 num_inference_steps=50, sigma_shift=5.0, verbose=False):
        self.pipe, self.norm = pipe, normalizer
        self.device, self.dtype = device, dtype
        self.height, self.width = height, width
        self.num_inference_steps, self.sigma_shift = num_inference_steps, sigma_shift
        self.verbose = verbose

    @classmethod
    def from_pretrained(cls, ckpt, wan_root, stats_json, device="cuda", dtype=torch.bfloat16, **kw):
        """ckpt: droid_wm_fullft_s64730.safetensors (full weights, no LoRA).
        wan_root: Wan2.2-TI2V-5B directory (provides the DiT architecture and Wan2.2_VAE.pth;
        every DiT weight is then overwritten by ckpt)."""
        torch.set_grad_enabled(False)
        # Full fine-tune checkpoint: LoRA injection and key mapping are identities.
        base.inject_lora = lambda dit, *a, **k: dit
        base.map_lora_keys = lambda sd: sd
        args = types.SimpleNamespace(ckpt=ckpt, model_root=wan_root, action_dim=7, lora_rank=32)
        pipe = base.build_pipe(args=args, device=device, dtype=dtype)
        return cls(pipe, ActionNormalizer(stats_json), device, dtype, **kw)

    @torch.inference_mode()
    def predict(self, obs_image, raw_actions, seed=0, normalized=False):
        """obs_image: PIL / uint8 array from a DROID exterior camera (any size, cropped to 4:3).
        raw_actions: (16, 7) rows as defined in droid_wm/actions.py (or already-normalised
        rows with normalized=True). Returns 16 PIL images (640x480): predicted frames 1..16."""
        obs = preprocess_image(obs_image, self.width, self.height)
        act = np.asarray(raw_actions, np.float32) if normalized else self.norm(raw_actions)
        act = torch.from_numpy(act)
        out = io.StringIO()
        with (contextlib.nullcontext() if self.verbose else contextlib.redirect_stdout(out)):
            lat_v, _, _ = base.sample(
                pipe=self.pipe, task="action_to_all", observation_image=obs, actions=act,
                gt_video=None, gt_contact=None, gt_force=None,
                height=self.height, width=self.width, num_frames=N_FUTURE + 1,
                num_inference_steps=self.num_inference_steps, sigma_shift=self.sigma_shift,
                seed=seed, device=self.device, dtype=self.dtype)
            frames = self.pipe.vae_output_to_video(self.pipe.vae.decode(lat_v, device=self.device))
        torch.cuda.empty_cache()
        return frames[1:]          # frames[0] is the reconstructed observation

    def rollout(self, obs_image, raw_actions, seed=0):
        """Autoregressive rollout: raw_actions (16*K, 7). Each chunk is conditioned on the
        last predicted frame of the previous chunk. Returns 16*K frames.
        Quality degrades with K (see README, 'Long horizons')."""
        raw_actions = np.asarray(raw_actions, np.float32)
        if raw_actions.ndim != 2 or raw_actions.shape[0] % N_FUTURE:
            raise ValueError(f"want (16*K, 7) actions, got {raw_actions.shape}")
        frames, obs = [], obs_image
        for k in range(raw_actions.shape[0] // N_FUTURE):
            chunk = self.predict(obs, raw_actions[k * N_FUTURE:(k + 1) * N_FUTURE], seed=seed + k)
            frames += chunk
            obs = chunk[-1]
        return frames
