# diffsynth/diffusion/flow_match_acwm_lv.py
"""
ACWM FlowMatch SFT loss with n-frame history conditioning (long-horizon variant).

Design (Plan 2: "flexible input" -- only observation RGB + action deltas):
    video   : the first `n` latent frames are CLEAN history conditions
              (VAE-encoded per-frame history images, written over the noised
              latents), and are NOT supervised.
    contact : the first `n - 1` latent frames are DUMMY zero padding inserted by
    force     ACWMInputVideoPhysicalEmbedder purely to keep the temporal length
              aligned across modalities. They carry no information and are NOT
              supervised. Latent frame `n - 1` onwards is the real
              (obs + future) physical sequence and IS supervised -- exactly as
              in the original single-observation setup.

Latent layout (n = n_history, num_frames = 17 = 1 obs + 16 future pixel frames):

    index :        0        1      ...     n-2   |   n-1   |   n    n+1  n+2  n+3
    video : [ hist_0   hist_1    ...   hist_n-2 | obs_lat |  fut_0 ... fut_3    ]
            \________________ clean condition ____________/ \___ supervised ___/
                              (n frames, sliced off)               (4 frames)

    phys  : [ zeros    zeros     ...    zeros   | obs_lat |  fut_0 ... fut_3    ]
            \_______ dummy pad (n-1, sliced) ___/\_________ supervised ________/
                                                            (5 frames)

Total latent length = n + 4 for all three modalities.
n = 1 degenerates EXACTLY to the original behaviour (video: slice 1 -> 4 frames
supervised; physical: slice 0 -> 5 frames supervised).

Expected inputs:
    input_latents_video / contact / force : (B, C, n+4, H, W)
    context                              : action embedding at DiT hidden dim
    first_frame_latents_video            : (B, C, n, H, W)  REQUIRED
        (alias `first_frame_latents` accepted for backward compatibility)
    first_frame_latents_contact / force  : (B, C, n, H, W)  OPTIONAL
        Unused under Plan 2 (always None). Wired up so that Plan 1
        (conditioning on previously generated physical frames) only requires
        the embedder to start emitting them -- no change needed here.
"""

import torch
import torch.nn.functional as F

from .base_pipeline import BasePipeline


def _apply_history_condition(latents, history, name):
    """
    Write history condition into latents[:, :, :n] if it exists.

    Args:
        latents: (B, C, T, H, W) -- noised latents, modified IN PLACE.
        history: (B, C, n, H, W) or None -- clean history latents.

    Returns:
        n (int) if applied, else None.
    """
    if history is None:
        return None

    if history.ndim != 5:
        raise ValueError(
            f"{name} must be 5-D (B, C, n, H, W), got shape {history.shape}."
        )

    n = history.shape[2]

    if n < 1:
        raise ValueError(
            f"{name} must contain at least 1 history frame "
            f"(n=1 means observation only), got n={n}."
        )

    if n > latents.shape[2]:
        raise ValueError(
            f"{name} has n={n} history frames but latents only have "
            f"T={latents.shape[2]} frames."
        )

    expected_shape = latents[:, :, :n].shape

    if history.shape != expected_shape:
        raise ValueError(
            f"{name} shape {history.shape} does not match "
            f"expected history latent shape {expected_shape}."
        )

    latents[:, :, :n] = history.to(
        dtype=latents.dtype,
        device=latents.device,
    )

    return n


def _check_shape(pred, target, name):
    if pred.shape != target.shape:
        raise ValueError(
            f"{name}: pred shape {pred.shape} does not match "
            f"target shape {target.shape}"
        )


def FlowMatchACWMSFTLoss(
    pipe: BasePipeline,
    **inputs,
):
    # ------------------------------------------------------------
    # 0. Sample timestep
    # ------------------------------------------------------------
    pipe.scheduler.set_timesteps(1000, training=True)

    num_timesteps = len(pipe.scheduler.timesteps)

    max_timestep_boundary = int(
        inputs.get("max_timestep_boundary", 1.0) * num_timesteps
    )
    min_timestep_boundary = int(
        inputs.get("min_timestep_boundary", 0.0) * num_timesteps
    )

    max_timestep_boundary = min(max_timestep_boundary, num_timesteps)
    min_timestep_boundary = max(min_timestep_boundary, 0)

    if max_timestep_boundary <= min_timestep_boundary:
        max_timestep_boundary = min_timestep_boundary + 1

    timestep_id = torch.randint(
        min_timestep_boundary,
        max_timestep_boundary,
        (1,),
        device=pipe.scheduler.timesteps.device,
    )

    timestep = pipe.scheduler.timesteps[timestep_id]

    timestep_model = timestep.to(dtype=pipe.torch_dtype, device=pipe.device)

    # ------------------------------------------------------------
    # 1. Sample noise for each modality
    # ------------------------------------------------------------
    noise_scale = inputs.get("noise_scale", 1.0)

    noise_video = torch.randn_like(inputs["input_latents_video"]) * noise_scale
    noise_contact = torch.randn_like(inputs["input_latents_contact"]) * noise_scale
    noise_force = torch.randn_like(inputs["input_latents_force"]) * noise_scale

    # ------------------------------------------------------------
    # 2. Add noise separately
    # ------------------------------------------------------------
    inputs["latents_video"] = pipe.scheduler.add_noise(
        inputs["input_latents_video"],
        noise_video,
        timestep,
    )

    inputs["latents_contact"] = pipe.scheduler.add_noise(
        inputs["input_latents_contact"],
        noise_contact,
        timestep,
    )

    inputs["latents_force"] = pipe.scheduler.add_noise(
        inputs["input_latents_force"],
        noise_force,
        timestep,
    )

    # ------------------------------------------------------------
    # 3. Training targets separately
    # ------------------------------------------------------------
    target_video = pipe.scheduler.training_target(
        inputs["input_latents_video"],
        noise_video,
        timestep,
    )

    target_contact = pipe.scheduler.training_target(
        inputs["input_latents_contact"],
        noise_contact,
        timestep,
    )

    target_force = pipe.scheduler.training_target(
        inputs["input_latents_force"],
        noise_force,
        timestep,
    )

    # ------------------------------------------------------------
    # 4. History conditioning
    #    video   : required, n clean frames written over the noised latents
    #    contact : optional (unused under Plan 2)
    #    force   : optional (unused under Plan 2)
    # ------------------------------------------------------------
    video_history = inputs.get("first_frame_latents_video", None)

    # Backward compatibility with original TI2V naming.
    if video_history is None:
        video_history = inputs.get("first_frame_latents", None)

    if video_history is None:
        raise ValueError(
            "ACWM requires video history conditioning. Expected "
            "`first_frame_latents_video` (B, C, n, H, W) or the legacy alias "
            "`first_frame_latents` in inputs."
        )

    contact_history = inputs.get("first_frame_latents_contact", None)
    force_history = inputs.get("first_frame_latents_force", None)

    n_video = _apply_history_condition(
        inputs["latents_video"],
        video_history,
        "first_frame_latents_video",
    )

    n_contact = _apply_history_condition(
        inputs["latents_contact"],
        contact_history,
        "first_frame_latents_contact",
    )

    n_force = _apply_history_condition(
        inputs["latents_force"],
        force_history,
        "first_frame_latents_force",
    )

    # All modalities must agree on n_history: the DiT computes
    # prefix_frames = f - num_action_latents from a single shared grid_size,
    # so a per-modality mismatch would silently misalign the action
    # cross-attention.
    for name, n_other in (("contact", n_contact), ("force", n_force)):
        if n_other is not None and n_other != n_video:
            raise ValueError(
                f"n_history mismatch: video has {n_video} history frames but "
                f"{name} has {n_other}. All modalities must share the same "
                f"n_history."
            )

    # Dummy zero padding inserted ahead of the physical sequences to keep the
    # temporal length aligned with video. Not supervised.
    physical_pad = n_video - 1

    # ------------------------------------------------------------
    # 5. Call model
    # ------------------------------------------------------------
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    pred = pipe.model_fn(
        **models,
        **inputs,
        timestep=timestep_model,
    )

    if not isinstance(pred, dict):
        raise TypeError(
            "FlowMatchACWMSFTLoss expects model_fn to return a dict: "
            "{'video': pred_video, 'contact': pred_contact, 'force': pred_force}. "
            f"Got {type(pred)} instead."
        )

    required_keys = ("video", "contact", "force")
    for key in required_keys:
        if key not in pred:
            raise KeyError(
                f"model_fn output is missing key '{key}'. "
                f"Expected keys: {required_keys}. Got keys: {list(pred.keys())}."
            )

    pred_video = pred["video"]
    pred_contact = pred["contact"]
    pred_force = pred["force"]

    # Full-sequence shape check.
    _check_shape(pred_video, target_video, "video")
    _check_shape(pred_contact, target_contact, "contact")
    _check_shape(pred_force, target_force, "force")

    # ------------------------------------------------------------
    # 6. Drop unsupervised prefix frames from the loss.
    #    NOTE the asymmetry: video drops n frames (real clean conditions),
    #    physical drops n-1 frames (dummy padding only). Under Plan 2 the
    #    physical observation latent at index n-1 REMAINS supervised, which
    #    reproduces the original single-observation behaviour exactly.
    # ------------------------------------------------------------
    pred_video = pred_video[:, :, n_video:]
    target_video = target_video[:, :, n_video:]

    if n_contact is not None:
        # Plan 1 path: contact history was supplied as a clean condition.
        contact_skip = n_contact
    else:
        contact_skip = physical_pad

    if n_force is not None:
        force_skip = n_force
    else:
        force_skip = physical_pad

    if contact_skip > 0:
        pred_contact = pred_contact[:, :, contact_skip:]
        target_contact = target_contact[:, :, contact_skip:]

    if force_skip > 0:
        pred_force = pred_force[:, :, force_skip:]
        target_force = target_force[:, :, force_skip:]

    # Shape check after prefix slicing.
    _check_shape(pred_video, target_video, "video_after_history_slicing")
    _check_shape(pred_contact, target_contact, "contact_after_history_slicing")
    _check_shape(pred_force, target_force, "force_after_history_slicing")

    if pred_video.shape[2] < 1:
        raise ValueError(
            f"No video frames left to supervise after slicing off "
            f"{n_video} history frames (total T={target_video.shape[2] + n_video})."
        )

    # ------------------------------------------------------------
    # 7. Loss weighting
    # ------------------------------------------------------------
    lambda_video = inputs.get("lambda_video", 1.0)
    lambda_contact = inputs.get("lambda_contact", 1.0)
    lambda_force = inputs.get("lambda_force", 1.0)

    flow_w = inputs.get("flow_weight", None)
    alpha = inputs.get("flow_alpha", 5.0)

    if flow_w is not None:
        # flow_weight comes from the dataset built over the ORIGINAL 17-frame
        # (1 obs + 16 future) clip, so it always has T_lat = 5 regardless of
        # n_history: the extra history frames are encoded separately and have no
        # flow weight. Dropping its observation frame leaves 4 frames, which
        # matches pred_video after slicing.
        fw = flow_w.to(device=pred_video.device, dtype=torch.float32)  # (B,1,T,h,w)
        fw = fw[:, :, 1:]

        if fw.shape[2] != pred_video.shape[2]:
            raise ValueError(
                f"flow_weight temporal length {fw.shape[2]} does not match "
                f"supervised video length {pred_video.shape[2]}. flow_weight is "
                f"expected to cover the 17-frame base clip (T_lat=5) only; check "
                f"that the dataset was not changed to include history frames."
            )

        w = 1.0 + alpha * fw
        se = (pred_video.float() - target_video.float()) ** 2
        loss_video = (w * se).sum() / w.expand_as(se).sum()
    else:
        loss_video = F.mse_loss(pred_video.float(), target_video.float())

    loss_contact = F.mse_loss(
        pred_contact.float(),
        target_contact.float(),
    )

    loss_force = F.mse_loss(
        pred_force.float(),
        target_force.float(),
    )

    loss = (
        lambda_video * loss_video
        + lambda_contact * loss_contact
        + lambda_force * loss_force
    )

    loss = loss * pipe.scheduler.training_weight(timestep)

    loss._acwm_stats = {
        "loss_video": loss_video.detach(),
        "loss_contact": loss_contact.detach(),
        "loss_force": loss_force.detach(),
        "timestep": float(timestep.detach().float().mean().cpu()),
        "n_history": n_video,
        "supervised_video_frames": pred_video.shape[2],
        "supervised_physical_frames": pred_contact.shape[2],
    }
    return loss