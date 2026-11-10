# diffsynth/diffusion/loss_acwm.py

import torch
import torch.nn.functional as F

from .base_pipeline import BasePipeline


def _apply_first_frame_condition(latents, first, name):
    """
    Write first-frame condition into latents[:, :, 0:1] if it exists.

    Args:
        latents: (B, C, T, H, W)
        first:   (B, C, 1, H, W) or None
    """
    if first is None:
        return False

    expected_shape = latents[:, :, 0:1].shape

    if first.shape != expected_shape:
        raise ValueError(
            f"{name} shape {first.shape} does not match "
            f"expected first-frame latent shape {expected_shape}. "
            f"Expected shape should be (B, C, 1, H, W)."
        )

    latents[:, :, 0:1] = first.to(
        dtype=latents.dtype,
        device=latents.device,
    )

    return True


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
    """
    ACWM FlowMatch SFT loss.

    Expected inputs:
        input_latents_video:   (B, C, T, H, W)
        input_latents_contact: (B, C, T, H, W)
        input_latents_force:   (B, C, T, H, W)

        context:
            action embedding, already projected to DiT hidden dim.

        first_frame_latents_video / first_frame_latents:
            required video first-frame condition, shape (B, C, 1, H, W).

        first_frame_latents_contact:
            optional contact first-frame condition, shape (B, C, 1, H, W).

        first_frame_latents_force:
            optional force first-frame condition, shape (B, C, 1, H, W).
    """

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
    # 4. First-frame conditioning
    # Video first frame is required.
    # Contact / force first frames are optional.
    # ------------------------------------------------------------
    video_first = inputs.get("first_frame_latents_video", None)

    # Backward compatibility with original TI2V naming.
    if video_first is None:
        video_first = inputs.get("first_frame_latents", None)

    if video_first is None:
        raise ValueError(
            "ACWM requires video first-frame conditioning. "
            "Expected `first_frame_latents_video` or `first_frame_latents` in inputs."
        )

    contact_first = inputs.get("first_frame_latents_contact", None)
    force_first = inputs.get("first_frame_latents_force", None)

    has_video_first = _apply_first_frame_condition(
        inputs["latents_video"],
        video_first,
        "first_frame_latents_video",
    )

    has_contact_first = _apply_first_frame_condition(
        inputs["latents_contact"],
        contact_first,
        "first_frame_latents_contact",
    )

    has_force_first = _apply_first_frame_condition(
        inputs["latents_force"],
        force_first,
        "first_frame_latents_force",
    )

    # ------------------------------------------------------------
    # 5. Call model
    # ------------------------------------------------------------
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    # print("[DEBUG] timestep_model device:", timestep_model.device, "dtype:", timestep_model.dtype)
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
    # 6. Remove conditioned first frame from loss.
    # If a modality's first frame is used as condition, do not supervise it.
    # ------------------------------------------------------------
    if has_video_first:
        pred_video = pred_video[:, :, 1:]
        target_video = target_video[:, :, 1:]

    if has_contact_first:
        pred_contact = pred_contact[:, :, 1:]
        target_contact = target_contact[:, :, 1:]

    if has_force_first:
        pred_force = pred_force[:, :, 1:]
        target_force = target_force[:, :, 1:]

    # Shape check after first-frame slicing.
    _check_shape(pred_video, target_video, "video_after_first_frame_slicing")
    _check_shape(pred_contact, target_contact, "contact_after_first_frame_slicing")
    _check_shape(pred_force, target_force, "force_after_first_frame_slicing")

    # ------------------------------------------------------------
    # 7. Loss weighting
    # ------------------------------------------------------------
    lambda_video = inputs.get("lambda_video", 1.0)
    lambda_contact = inputs.get("lambda_contact", 1.0)
    lambda_force = inputs.get("lambda_force", 1.0)

    # loss_video = F.mse_loss(
    #     pred_video.float(),
    #     target_video.float(),
    # )
    flow_w = inputs.get("flow_weight", None)
    alpha = inputs.get("flow_alpha", 5.0)

    if flow_w is not None:
        fw = flow_w.to(device=pred_video.device, dtype=torch.float32)  # (B,1,T,h,w)
        if has_video_first:
            fw = fw[:, :, 1:]
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
    }
    return loss
    