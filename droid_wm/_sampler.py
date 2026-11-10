"""Vendored sampler (unchanged copy of infer_acwm_mot_videoonly_multigpu.py from the
research repo; only the grasping-dataset import was made optional). droid_wm.model
uses build_pipe / sample / the VAE from here."""
#!/usr/bin/env python3

"""
infer_acwm_mot_videoonly_multigpu.py

Inference for ablation Arm B -- the video-only ACWM MoT model, trained with
no physical supervision. Copied from infer_acwm_mot_forcing_multigpu.py and
reduced to a single modality; the sharding, LoRA, scheduler and output layout
are unchanged, so eval_acwm_v3.py reads both arms the same way.

Task
----
action_to_all
    condition: observation + action
    generate : video

That is the only task this arm was trained on, and the only one it can run:
the model has no contact/force branches to condition on or predict. The other
three forcing tasks all involve a physical modality.

Important
---------
1. Action is always a condition here, so the action encoder's null token is
   never used at inference (it is still in the checkpoint, untrained).
2. The observation latent at frame 0 stays fixed across denoising steps.
3. LoRA is injected BEFORE checkpoint loading.
4. No contact/force VAE is loaded. Asking for one is a configuration error,
   not something to silently ignore -- Arm B must not accidentally be
   evaluated with physical outputs copied from somewhere else.
"""

import argparse
import inspect
import json
import os
import shutil

import numpy as np
import torch
from PIL import Image

from diffsynth.pipelines.wan_video import ModelConfig
from diffsynth.models.action_encoder import ActionFFNEncoder
from diffsynth.diffusion.flow_match_acwm_videoonly import (
    TASK_SPECS,
    task_uses_action,
)

try:  # only needed by the original grasping eval main(); not used by droid_wm
    from acwm_dataset import ACWMDataset, _frame_stem_to_npy
except ImportError:
    ACWMDataset = _frame_stem_to_npy = None


# ======================================================================
# Tasks
# ======================================================================

# Only the task this arm was trained on. The three physical forcing tasks are
# not merely untrained here -- the model has no contact/force branches at all.
SUPPORTED_TASKS = set(TASK_SPECS)


def task_flags(task):
    """
    The video-only arm has exactly one task. Keeping the function (rather
    than inlining its result) means sample() and main() below stay
    line-for-line comparable with the forcing script they were copied from.
    """
    if task not in TASK_SPECS:
        raise ValueError(
            f"{task!r} is not available in the video-only arm; "
            f"it was trained on {sorted(TASK_SPECS)} only. The other forcing "
            f"tasks all condition on or predict a physical modality."
        )
    # All six flags, with the physical ones pinned False, so every
    # `if flags["condition_contact"]` / `if flags["target_force"]` branch
    # inherited from the forcing script simply does not fire. That keeps this
    # copy diffable against the original instead of restructuring sample().
    return {
        "condition_video": False,
        "condition_contact": False,
        "condition_force": False,
        "target_video": True,
        "target_contact": False,
        "target_force": False,
    }


def generated_modalities(task):
    task_flags(task)
    return ["video"]


def condition_modalities(task):
    task_flags(task)
    return ["observation", "action"]


# ======================================================================
# LoRA
# ======================================================================

# Must match sbatch_abl_videoonly.sh exactly. Ten modules per block: the
# video self-attention, the shared cross-attention k/v, the video-specific
# cross-attention q/o, and the two video FFN layers. cross_attn.q and
# cross_attn.o are deliberately absent -- the MoT block deep-copies them into
# cross_q_video / cross_o_video and never calls the originals, so LoRA there
# would be dead weight (and would break DDP during training).
MOT_LORA_TARGETS = ",".join([
    "self_attn_video.q",
    "self_attn_video.k",
    "self_attn_video.v",
    "self_attn_video.o",

    "cross_attn.k",
    "cross_attn.v",

    "cross_q_video",
    "cross_o_video",

    "ffn_video.0",
    "ffn_video.2",
])


def inject_lora(
    model,
    target_modules,
    lora_rank,
    upcast_dtype,
):
    from peft import LoraConfig, inject_adapter_in_model

    targets = [
        x.strip()
        for x in target_modules.split(",")
        if x.strip()
    ]

    cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=targets,
    )

    model = inject_adapter_in_model(
        cfg,
        model,
    )

    if upcast_dtype is not None:
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.to(upcast_dtype)

    return model


def map_lora_keys(sd):
    """
    Normalize old/new PEFT naming.
    """

    out = {}

    for k, v in sd.items():

        k = k.replace(
            "lora_A.weight",
            "lora_A.default.weight",
        )

        k = k.replace(
            "lora_B.weight",
            "lora_B.default.weight",
        )

        out[k] = v

    return out


# ======================================================================
# Physical VAEs
# ======================================================================

def build_pipe(
    args,
    device,
    dtype,
):

    # --------------------------------------------------------------
    # Wan base model
    # --------------------------------------------------------------

    dit_paths = [
        os.path.join(
            args.model_root,
            f"diffusion_pytorch_model-0000{i}-of-00003.safetensors",
        )
        for i in (1, 2, 3)
    ]

    vae_path = os.path.join(
        args.model_root,
        "Wan2.2_VAE.pth",
    )

    model_configs = [
        ModelConfig(path=dit_paths),
        ModelConfig(path=vae_path),
    ]

    # --------------------------------------------------------------
    # ALWAYS forcing MoT pipeline.
    #
    # No --use_mot flag. This prevents accidentally loading the
    # baseline/shared pipeline.
    # --------------------------------------------------------------

    from diffsynth.pipelines.wan_video_acwm_mot_videoonly import (
        ACWMWanVideoPipelineMoTForcing,
    )

    print(
        "[load] initializing "
        "ACWMWanVideoPipelineMoTForcing..."
    )

    pipe = ACWMWanVideoPipelineMoTForcing.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=None,
        audio_processor_config=None,
        redirect_common_files=False,
    )

    print(
        "[load] pipe:",
        type(pipe).__name__,
    )

    print(
        "[load] dit:",
        type(pipe.dit).__name__,
    )

    print(
        "[load] block0:",
        type(pipe.dit.blocks[0]).__name__,
    )

    # --------------------------------------------------------------
    # Action encoder
    # --------------------------------------------------------------

    pipe.action_encoder = ActionFFNEncoder(
        action_dim=args.action_dim,
        embed_dim=pipe.dit.dim,
    ).to(
        device=device,
        dtype=dtype,
    )

    pipe.action_proj = None

    # --------------------------------------------------------------
    # No physical VAEs.
    #
    # Set them to None rather than leaving the attributes unset: anything
    # that reaches for a physical VAE in this arm is a bug, and an
    # AttributeError on None fails at the point of misuse instead of
    # silently falling through to a stale module attribute.
    # --------------------------------------------------------------
    pipe.contact_vae = None
    pipe.force_vae = None

    # --------------------------------------------------------------
    # CRITICAL:
    # Inject LoRA BEFORE loading checkpoint.
    # --------------------------------------------------------------

    print("[load] injecting MoT LoRA...")

    pipe.dit = inject_lora(
        pipe.dit,
        MOT_LORA_TARGETS,
        args.lora_rank,
        dtype,
    )

    # --------------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------------

    from safetensors.torch import load_file

    print(
        f"[load] checkpoint: {args.ckpt}"
    )

    sd = load_file(
        args.ckpt,
        device="cpu",
    )

    sd = map_lora_keys(sd)

    # Training ModelLogger may remove "pipe.".
    cleaned = {}

    for k, v in sd.items():

        if k.startswith("pipe."):
            k = k[len("pipe."):]

        cleaned[k] = v

    sd = cleaned

    # --------------------------------------------------------------
    # Action encoder checkpoint
    # --------------------------------------------------------------

    ae_sd = {
        k[len("action_encoder."):]: v
        for k, v in sd.items()
        if k.startswith("action_encoder.")
    }

    missing_ae, unexpected_ae = (
        pipe.action_encoder.load_state_dict(
            ae_sd,
            strict=False,
        )
    )

    print(
        f"[load] action_encoder: "
        f"{len(ae_sd)} keys, "
        f"missing={len(missing_ae)}, "
        f"unexpected={len(unexpected_ae)}"
    )

    if unexpected_ae:
        raise RuntimeError(
            "Unexpected action_encoder keys: "
            f"{unexpected_ae[:20]}"
        )

    # load_state_dict(strict=False) silently tolerates MISSING keys, so a
    # checkpoint predating the null token would leave it at random init. The
    # no-action tasks would then run on noise and produce garbage with no
    # warning. Record it here and fail loudly at use (see sample()), rather
    # than blocking action_to_all / contact_to_video_force, which do not need
    # the null token at all.
    null_token_loaded = "null_token" not in missing_ae
    pipe.action_encoder._null_token_loaded = null_token_loaded

    if not null_token_loaded:
        print(
            "[load] WARNING: action_encoder.null_token is NOT in this "
            "checkpoint (it predates the no-action tasks). "
            "video_to_contact_force and video_contact_to_force will refuse "
            "to run."
        )

    # --------------------------------------------------------------
    # DiT checkpoint
    # --------------------------------------------------------------

    dit_sd = {
        k[len("dit."):]: v
        for k, v in sd.items()
        if k.startswith("dit.")
    }

    model_keys = set(
        pipe.dit.state_dict().keys()
    )

    ckpt_keys = set(
        dit_sd.keys()
    )

    matched = model_keys & ckpt_keys
    bad = ckpt_keys - model_keys

    print(
        f"[load] dit key match: "
        f"{len(matched)}/{len(dit_sd)} "
        f"("
        f"{100.0 * len(matched) / max(1, len(dit_sd)):.2f}%"
        f")"
    )

    if bad:

        print(
            "[ERROR] first unexpected checkpoint keys:"
        )

        for k in sorted(bad)[:20]:
            print("   ", k)

        raise RuntimeError(
            f"{len(bad)} checkpoint DiT keys "
            f"do not exist in inference model."
        )

    missing_dit, unexpected_dit = (
        pipe.dit.load_state_dict(
            dit_sd,
            strict=False,
        )
    )

    print(
        f"[load] dit: "
        f"{len(dit_sd)} keys, "
        f"missing={len(missing_dit)}, "
        f"unexpected={len(unexpected_dit)}"
    )

    # Frozen Wan base weights are expected missing from the
    # partial training checkpoint.
    if unexpected_dit:
        raise RuntimeError(
            "Unexpected DiT checkpoint keys after load: "
            f"{unexpected_dit[:20]}"
        )

    pipe.dit.eval()
    pipe.action_encoder.eval()

    pipe.dit.requires_grad_(False)
    pipe.action_encoder.requires_grad_(False)

    print(
        "[load] forcing checkpoint loaded successfully."
    )


    print("[DEBUG TYPES]")
    print("pipe.vae        =", type(pipe.vae))
    print("pipe.dit        =", type(pipe.dit))

    return pipe


# ======================================================================
# Video condition
# ======================================================================

@torch.no_grad()
def encode_video_condition(
    pipe,
    observation_image,
    future_video,
    width,
    height,
    device,
    dtype,
):
    """
    Encode observation + 16 future GT RGB frames.

    Returns expected shape:
        B,48,5,30,40
    for 17 x 480 x 640.
    """

    all_frames = [
        observation_image,
        *list(future_video),
    ]

    tensors = []

    for frame in all_frames:

        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(
                np.asarray(frame)
            )

        frame = frame.resize(
            (width, height)
        )

        x = pipe.preprocess_image(
            frame
        ).transpose(0, 1)

        tensors.append(x)

    video_tensor = torch.cat(
        tensors,
        dim=1,
    )

    latent = pipe.vae.encode(
        [video_tensor],
        device=device,
    )

    latent = latent.to(
        device=device,
        dtype=dtype,
    )

    return latent


# ======================================================================
# Physical condition
# ======================================================================

def prepare_physical_tensor(
    x,
    channels,
    device,
    dtype,
):
    """
    Dataset physical modality -> B,C,T,H,W.

    Supports:
        C,T,H,W
        T,C,H,W
        B,C,T,H,W
        B,T,C,H,W

    IMPORTANT:
    ACWMDataset already performs training-time physical normalization.
    Do NOT normalize item["contact"] / item["force"] again here.
    """

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    if x.ndim == 4:

        if x.shape[0] == channels:

            # C,T,H,W
            x = x.unsqueeze(0)

        elif x.shape[1] == channels:

            # T,C,H,W -> B,C,T,H,W
            x = x.permute(
                1,
                0,
                2,
                3,
            ).unsqueeze(0)

        else:

            raise ValueError(
                f"Cannot infer physical layout "
                f"from {tuple(x.shape)} "
                f"with channels={channels}"
            )

    elif x.ndim == 5:

        if x.shape[1] == channels:

            # B,C,T,H,W
            pass

        elif x.shape[2] == channels:

            # B,T,C,H,W -> B,C,T,H,W
            x = x.permute(
                0,
                2,
                1,
                3,
                4,
            )

        else:

            raise ValueError(
                f"Cannot infer physical layout "
                f"from {tuple(x.shape)} "
                f"with channels={channels}"
            )

    else:

        raise ValueError(
            "Physical tensor must be 4D or 5D, "
            f"got {tuple(x.shape)}"
        )

    return x.to(
        device=device,
        dtype=dtype,
    )


def unwrap_latent(z):
    """
    Robustly unwrap common encoder return types.
    """

    if torch.is_tensor(z):
        return z

    if isinstance(z, (tuple, list)):

        if len(z) == 0:
            raise RuntimeError(
                "Physical VAE returned empty tuple/list."
            )

        return unwrap_latent(z[0])

    if isinstance(z, dict):

        for key in (
            "latent",
            "latents",
            "z",
            "sample",
        ):
            if key in z:
                return unwrap_latent(
                    z[key]
                )

    if hasattr(z, "latent_dist"):
        dist = z.latent_dist

        if hasattr(dist, "mode"):
            return dist.mode()

        return dist.sample()

    if hasattr(z, "mode"):
        try:
            out = z.mode()
            if torch.is_tensor(out):
                return out
        except Exception:
            pass

    if hasattr(z, "sample"):
        try:
            out = z.sample()
            if torch.is_tensor(out):
                return out
        except Exception:
            pass

    raise TypeError(
        f"Unsupported physical VAE output type: "
        f"{type(z)}"
    )


@torch.no_grad()
def encode_physical_condition(
    vae,
    x,
    channels,
    device,
    dtype,
    name,
):
    """
    Encode normalized physical modality using ContactVAE / ForceVAE.

    ACWMDataset already returns normalized physical tensors.

    Input supported:
        (C,T,H,W)
        (T,C,H,W)
        (B,C,T,H,W)
        (B,T,C,H,W)

    PhysicalVAE inherits WanVideoVAE.encode(), whose API is:

        vae.encode(
            videos: List[Tensor(C,T,H,W)],
            device=device,
        )

    Returns:
        (B,48,T_lat,H_lat,W_lat)
    """

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    # ------------------------------------------------------------
    # Normalize layout to B,C,T,H,W
    # ------------------------------------------------------------

    if x.ndim == 4:

        if x.shape[0] == channels:
            # C,T,H,W
            x = x.unsqueeze(0)

        elif x.shape[1] == channels:
            # T,C,H,W -> B,C,T,H,W
            x = x.permute(
                1,
                0,
                2,
                3,
            ).unsqueeze(0)

        else:
            raise ValueError(
                f"{name}: cannot infer layout "
                f"from shape={tuple(x.shape)}, "
                f"channels={channels}"
            )

    elif x.ndim == 5:

        if x.shape[1] == channels:
            # already B,C,T,H,W
            pass

        elif x.shape[2] == channels:
            # B,T,C,H,W -> B,C,T,H,W
            x = x.permute(
                0,
                2,
                1,
                3,
                4,
            )

        else:
            raise ValueError(
                f"{name}: cannot infer layout "
                f"from shape={tuple(x.shape)}, "
                f"channels={channels}"
            )

    else:
        raise ValueError(
            f"{name}: expected 4D or 5D tensor, "
            f"got {tuple(x.shape)}"
        )

    x = x.to(
        device=device,
        dtype=dtype,
    )

    print(
        f"[condition] {name} input: "
        f"{tuple(x.shape)} "
        f"vae={type(vae).__name__}"
    )

    # ------------------------------------------------------------
    # CRITICAL:
    #
    # ContactVAE / ForceVAE inherit WanVideoVAE.encode().
    #
    # WanVideoVAE expects:
    #
    #     List[Tensor(C,T,H,W)]
    #
    # NOT a batched B,C,T,H,W tensor directly.
    # ------------------------------------------------------------

    videos = [
        x[b]
        for b in range(x.shape[0])
    ]

    z = vae.encode(
        videos,
        device=device,
    )

    # Depending on local implementation, encode may return:
    #
    #   Tensor(B,C,T,H,W)
    #
    # or
    #
    #   list[Tensor(C,T,H,W)]
    #
    # Normalize both.
    if isinstance(z, (list, tuple)):

        if len(z) == 0:
            raise RuntimeError(
                f"{name}: VAE returned empty latent list"
            )

        if torch.is_tensor(z[0]):

            if z[0].ndim == 4:
                z = torch.stack(
                    list(z),
                    dim=0,
                )

            else:
                z = z[0]

        else:
            raise TypeError(
                f"{name}: unsupported VAE latent "
                f"type inside list: {type(z[0])}"
            )

    if not torch.is_tensor(z):
        raise TypeError(
            f"{name}: VAE encode returned "
            f"{type(z)}"
        )

    z = z.to(
        device=device,
        dtype=dtype,
    )

    print(
        f"[condition] {name} latent: "
        f"{tuple(z.shape)}"
    )

    return z


# ======================================================================
# Action
# ======================================================================

@torch.no_grad()
def build_action_context(
    pipe,
    actions,
    device,
    dtype,
    drop_action=False,
):
    """
    IMPORTANT:

    Do NOT zero this context for the no-action tasks. Zeroing is not "no
    action": actions are normalized by (a - mean) / std, so a zero vector
    decodes to the dataset-mean action, which is a specific condition rather
    than an absent one. Pass drop_action=True instead, which emits the action
    encoder's learnable null token -- the same symbol those tasks were trained
    with.
    """

    actions = actions.to(
        device=device,
        dtype=dtype,
    )

    if actions.ndim == 2:
        actions = actions.unsqueeze(0)

    if drop_action:
        context = pipe.action_encoder(
            actions,
            drop_action=True,
        )
    else:
        context = pipe.action_encoder(
            actions
        )

    if pipe.action_proj is not None:
        context = pipe.action_proj(
            context
        )

    return context.to(
        device=device,
        dtype=dtype,
    )


# ======================================================================
# Forcing model_fn
# ======================================================================

def call_forcing_model_fn(
    pipe,
    *,
    lat_v,
    lat_c,
    lat_f,
    timestep_video,
    timestep_contact,
    timestep_force,
    context,
):
    """
    Call the forcing pipeline using per-modality timesteps.

    Supports a few local naming variations, but REQUIRES the forcing
    pipeline to expose independent modality timesteps.
    """

    sig = inspect.signature(
        pipe.model_fn
    )

    names = set(
        sig.parameters.keys()
    )

    kwargs = {
        "dit": pipe.dit,
        "latents_video": lat_v,
        "latents_contact": lat_c,
        "latents_force": lat_f,
        "context": context,
        "use_gradient_checkpointing": False,
    }

    # --------------------------------------------------------------
    # Variant A
    # timestep_video / timestep_contact / timestep_force
    # --------------------------------------------------------------

    if {
        "timestep_video",
        "timestep_contact",
        "timestep_force",
    }.issubset(names):

        kwargs.update({
            "timestep_video": timestep_video,
            "timestep_contact": timestep_contact,
            "timestep_force": timestep_force,
        })

    # --------------------------------------------------------------
    # Variant B
    # t_video / t_contact / t_force
    # --------------------------------------------------------------

    elif {
        "t_video",
        "t_contact",
        "t_force",
    }.issubset(names):

        kwargs.update({
            "t_video": timestep_video,
            "t_contact": timestep_contact,
            "t_force": timestep_force,
        })

    # --------------------------------------------------------------
    # Variant C
    # t_mod_video / t_mod_contact / t_mod_force
    # --------------------------------------------------------------

    elif {
        "t_mod_video",
        "t_mod_contact",
        "t_mod_force",
    }.issubset(names):

        kwargs.update({
            "t_mod_video": timestep_video,
            "t_mod_contact": timestep_contact,
            "t_mod_force": timestep_force,
        })

    else:

        raise RuntimeError(
            "\nForcing model_fn does not expose per-modality "
            "timesteps.\n"
            f"model_fn signature:\n{sig}\n\n"
            "Expected one of:\n"
            "  timestep_video/timestep_contact/timestep_force\n"
            "  t_video/t_contact/t_force\n"
            "  t_mod_video/t_mod_contact/t_mod_force\n"
        )

    # Only retain kwargs accepted by the function unless **kwargs exists.
    has_varkw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )

    if not has_varkw:

        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in names
        }

    return pipe.model_fn(
        **kwargs
    )


def normalize_prediction(pred):
    """
    Normalize model output to {"video": ...}.

    The forcing version accepted a 3-tuple or a 3-key dict. This arm's DiT
    returns {"video": ...} only, and anything carrying a contact/force key
    means the wrong model was loaded -- say so rather than quietly dropping
    the extra modalities and reporting video numbers for a physical model.
    """
    if isinstance(pred, dict):
        stray = {"contact", "force"} & set(pred)
        if stray:
            raise TypeError(
                f"model returned {sorted(stray)} -- this is the video-only "
                f"arm, so the checkpoint or pipeline module is wrong."
            )
        for key in ("video", "pred_video", "noise_video"):
            if key in pred:
                return {"video": pred[key]}

    if isinstance(pred, torch.Tensor):
        return {"video": pred}

    raise TypeError(
        f"Cannot interpret video-only model output. type={type(pred)}"
    )


# ======================================================================
# Scheduler helper
# ======================================================================

def scheduler_step(
    scheduler,
    prediction,
    timestep,
    latent,
):
    """
    Support scheduler.step returning either tensor or object.
    """

    out = scheduler.step(
        prediction,
        timestep,
        latent,
    )

    if torch.is_tensor(out):
        return out

    if hasattr(out, "prev_sample"):
        return out.prev_sample

    if isinstance(out, (tuple, list)):
        return out[0]

    raise TypeError(
        f"Unsupported scheduler.step output: "
        f"{type(out)}"
    )


# ======================================================================
# Sampling
# ======================================================================

@torch.no_grad()
def sample(
    pipe,
    task,
    observation_image,
    actions,
    gt_video,
    gt_contact,
    gt_force,
    height,
    width,
    num_frames,
    num_inference_steps,
    sigma_shift,
    seed,
    device,
    dtype,
):

    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task: {task}"
        )

    flags = task_flags(task)

    print()
    print("=" * 80)
    print(f"[task] {task}")

    print(
        "[task] condition:",
        condition_modalities(task),
    )

    print(
        "[task] generate:",
        generated_modalities(task),
    )

    print("=" * 80)

    # --------------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------------

    sch = pipe.scheduler

    sch.set_timesteps(
        num_inference_steps,
        denoising_strength=1.0,
        shift=sigma_shift,
    )

    # --------------------------------------------------------------
    # Latent shape
    # --------------------------------------------------------------

    length = (
        (num_frames - 1) // 4
        + 1
    )

    h = (
        height
        // pipe.vae.upsampling_factor
    )

    w = (
        width
        // pipe.vae.upsampling_factor
    )

    c = pipe.vae.model.z_dim

    shape = (
        1,
        c,
        length,
        h,
        w,
    )

    print(
        "[latent] shape:",
        shape,
    )

    # --------------------------------------------------------------
    # Noise
    # --------------------------------------------------------------

    lat_v = pipe.generate_noise(
        shape,
        seed=seed,
        rand_device="cpu",
    ).to(
        device=device,
        dtype=dtype,
    )

    # No contact/force noise. lat_v above drew from `seed`; the forcing
    # script's contact/force draws used seed+1 / seed+2 as independent
    # generators, so dropping them leaves lat_v bit-identical to Arm A's at
    # the same seed. Both arms therefore start denoising from the same video
    # noise, which is what makes a per-clip comparison meaningful.
    lat_c = None
    lat_f = None

    # --------------------------------------------------------------
    # Observation latent
    # --------------------------------------------------------------

    obs_img = observation_image.resize(
        (width, height)
    )

    obs_tensor = pipe.preprocess_image(
        obs_img
    ).transpose(0, 1)

    obs_lat = pipe.vae.encode(
        [obs_tensor],
        device=device,
    ).to(
        device=device,
        dtype=dtype,
    )

    print(
        "[condition] observation latent:",
        tuple(obs_lat.shape),
    )

    # --------------------------------------------------------------
    # Clean GT condition latents
    # --------------------------------------------------------------

    gt_video_lat = None
    gt_contact_lat = None
    gt_force_lat = None

    # GT video condition
    if flags["condition_video"]:

        if gt_video is None:
            raise ValueError(
                f"{task} requires item['video']."
            )

        print(
            "[condition] encoding GT video..."
        )

        gt_video_lat = encode_video_condition(
            pipe=pipe,
            observation_image=observation_image,
            future_video=gt_video,
            width=width,
            height=height,
            device=device,
            dtype=dtype,
        )

        print(
            "[condition] GT video latent:",
            tuple(gt_video_lat.shape),
        )

        lat_v = gt_video_lat.clone()

    else:

        # Video is generated.
        # Observation frame is always clean/fixed.
        lat_v[:, :, 0:1] = obs_lat

    # GT contact condition
    if flags["condition_contact"]:

        if gt_contact is None:
            raise ValueError(
                f"{task} requires item['contact']."
            )

        print(
            "[condition] encoding GT contact..."
        )

        gt_contact_lat = encode_physical_condition(
            vae=pipe.contact_vae,
            x=gt_contact,
            channels=2,
            device=device,
            dtype=dtype,
            name="contact",
        )

        if (
            gt_contact_lat.shape
            != lat_c.shape
        ):
            raise RuntimeError(
                "GT contact latent shape mismatch:\n"
                f"  GT     = {tuple(gt_contact_lat.shape)}\n"
                f"  target = {tuple(lat_c.shape)}"
            )

        lat_c = gt_contact_lat.clone()

    # Included for future forcing variants.
    if flags["condition_force"]:

        if gt_force is None:
            raise ValueError(
                f"{task} requires item['force']."
            )

        gt_force_lat = encode_physical_condition(
            vae=pipe.force_vae,
            x=gt_force,
            channels=6,
            device=device,
            dtype=dtype,
            name="force",
        )

        lat_f = gt_force_lat.clone()

    # --------------------------------------------------------------
    # ACTION CONDITION
    #
    # Active only for the action-driven tasks. The video-conditioned tasks
    # get the action encoder's null token instead.
    # --------------------------------------------------------------

    uses_action = task_uses_action(task)

    if not uses_action and not getattr(
        pipe.action_encoder,
        "_null_token_loaded",
        True,
    ):
        raise RuntimeError(
            f"Task '{task}' runs without an action and therefore needs the "
            f"trained action_encoder.null_token, but this checkpoint does not "
            f"contain it. Running anyway would feed random noise as the "
            f"action context. Use a checkpoint trained after the no-action "
            f"tasks were introduced."
        )

    context = build_action_context(
        pipe=pipe,
        actions=actions,
        device=device,
        dtype=dtype,
        drop_action=not uses_action,
    )

    print(
        "[action] context:",
        tuple(context.shape),
        "" if uses_action else "(NULL TOKEN - this task has no action)",
    )

    # --------------------------------------------------------------
    # Denoise
    # --------------------------------------------------------------

    for i, t in enumerate(
        sch.timesteps
    ):

        if torch.is_tensor(t):
            ts = t.reshape(1).to(
                device=device,
                dtype=dtype,
            )
        else:
            ts = torch.tensor(
                [t],
                device=device,
                dtype=dtype,
            )

        zero_ts = torch.zeros_like(
            ts
        )

        # ----------------------------------------------------------
        # CRITICAL forcing behavior:
        #
        # condition -> timestep 0
        # target    -> current timestep
        # ----------------------------------------------------------

        timestep_video = (
            zero_ts
            if flags["condition_video"]
            else ts
        )

        timestep_contact = (
            zero_ts
            if flags["condition_contact"]
            else ts
        )

        timestep_force = (
            zero_ts
            if flags["condition_force"]
            else ts
        )

        pred = call_forcing_model_fn(
            pipe,
            lat_v=lat_v,
            lat_c=lat_c,
            lat_f=lat_f,
            timestep_video=timestep_video,
            timestep_contact=timestep_contact,
            timestep_force=timestep_force,
            context=context,
        )

        pred = normalize_prediction(
            pred
        )

        # ----------------------------------------------------------
        # Update ONLY target modalities.
        # ----------------------------------------------------------

        if flags["target_video"]:

            lat_v = scheduler_step(
                sch,
                pred["video"],
                sch.timesteps[i],
                lat_v,
            )

        if flags["target_contact"]:

            lat_c = scheduler_step(
                sch,
                pred["contact"],
                sch.timesteps[i],
                lat_c,
            )

        if flags["target_force"]:

            lat_f = scheduler_step(
                sch,
                pred["force"],
                sch.timesteps[i],
                lat_f,
            )

        # ----------------------------------------------------------
        # Re-clamp clean conditions.
        # ----------------------------------------------------------

        if flags["condition_video"]:

            lat_v = gt_video_lat.clone()

        else:

            # Generated video always retains clean observation latent.
            lat_v[:, :, 0:1] = obs_lat

        if flags["condition_contact"]:

            lat_c = gt_contact_lat.clone()

        if flags["condition_force"]:

            lat_f = gt_force_lat.clone()

        if (
            i == 0
            or i == len(sch.timesteps) - 1
            or (i + 1) % 10 == 0
        ):

            print(
                f"[denoise] "
                f"{i + 1}/{len(sch.timesteps)} "
                f"t={float(t):.4f}"
            )

    return (
        lat_v,
        lat_c,
        lat_f,
    )


# ======================================================================
# Decode
# ======================================================================

@torch.no_grad()
def decode_and_save(
    pipe,
    lat_v,
    lat_c,
    lat_f,
    out_dir,
    device,
):

    gen_video_dir = os.path.join(
        out_dir,
        "gen_video",
    )

    gen_contact_dir = os.path.join(
        out_dir,
        "gen_contact",
    )

    gen_force_dir = os.path.join(
        out_dir,
        "gen_force",
    )

    os.makedirs(
        gen_video_dir,
        exist_ok=True,
    )

    os.makedirs(
        gen_contact_dir,
        exist_ok=True,
    )

    os.makedirs(
        gen_force_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Video
    # --------------------------------------------------------------

    frames = pipe.vae.decode(
        lat_v,
        device=device,
    )

    frames = pipe.vae_output_to_video(
        frames
    )

    for i, im in enumerate(frames):

        im.save(
            os.path.join(
                gen_video_dir,
                f"frame_{i:02d}.png",
            )
        )

    # No contact/force decode: this arm has no physical VAEs and produced no
    # physical latents. gen_contact/ and gen_force/ are created above but left
    # EMPTY on purpose -- eval_acwm_v3.py then reports video metrics only.
    # Writing zeros instead would look like a confident all-background
    # prediction and score as if the model had made one.

    # The forcing version also returned the physical frame count; there is no
    # physical output here and the caller never read it.
    return len(frames)


# ======================================================================
# Raw GT copy
# ======================================================================

def copy_gt(
    sample,
    source_root,
    out_dir,
):

    for sub in (
        "gt_video",
        "gt_contact",
        "gt_force",
    ):

        os.makedirs(
            os.path.join(
                out_dir,
                sub,
            ),
            exist_ok=True,
        )

    obs_png = sample[
        "observation_frame"
    ]

    mod_pngs = (
        [obs_png]
        + list(sample["frames"])
    )

    def mod_npy_path(
        png_rel,
        modality,
    ):

        stem = _frame_stem_to_npy(
            png_rel
        )

        images_dir = os.path.dirname(
            png_rel
        )

        episode_dir = os.path.dirname(
            images_dir
        )

        return os.path.join(
            episode_dir,
            "modalities",
            modality,
            f"{stem}.npy",
        )

    for i, png_rel in enumerate(
        mod_pngs
    ):

        # RGB
        src_png = os.path.join(
            source_root,
            png_rel,
        )

        if os.path.exists(src_png):

            shutil.copy(
                src_png,
                os.path.join(
                    out_dir,
                    "gt_video",
                    f"frame_{i:02d}.png",
                ),
            )

        # physical
        for modality, sub in (
            ("contact", "gt_contact"),
            ("force", "gt_force"),
        ):

            src_npy = os.path.join(
                source_root,
                mod_npy_path(
                    png_rel,
                    modality,
                ),
            )

            if os.path.exists(src_npy):

                shutil.copy(
                    src_npy,
                    os.path.join(
                        out_dir,
                        sub,
                        f"{i:02d}.npy",
                    ),
                )


# ======================================================================
# Main
# ======================================================================

def main():

    ap = argparse.ArgumentParser()

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    ap.add_argument(
        "--ckpt",
        required=True,
    )

    ap.add_argument(
        "--model_root",
        required=True,
    )

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    ap.add_argument(
        "--pipeline_json",
        required=True,
    )

    ap.add_argument(
        "--source_root",
        required=True,
    )

    ap.add_argument(
        "--physical_stats",
        required=True,
    )

    ap.add_argument(
        "--action_stats",
        required=True,
    )

    # --------------------------------------------------------------
    # Physical VAE
    # --------------------------------------------------------------

    ap.add_argument(
        # Accepted but unused, so one sbatch can drive both arms. Passing a
        # path is a mistake worth surfacing: it means the caller believes this
        # arm produces physical output.
        "--contact_vae_ckpt",
        default=None,
    )

    ap.add_argument(
        "--force_vae_ckpt",
        default=None,
    )

    # --------------------------------------------------------------
    # Architecture
    # --------------------------------------------------------------

    ap.add_argument(
        "--action_dim",
        type=int,
        default=7,
    )

    ap.add_argument(
        # Arm B trains at rank 88 (sbatch_abl_videoonly.sh): that is what
        # makes its trainable parameter count match Arm A's 221.17M given
        # this arm's 10 target modules per block. Loading an 88-rank
        # checkpoint into a rank-32 injection silently mismatches shapes.
        "--lora_rank",
        type=int,
        default=88,
    )

    # --------------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------------

    ap.add_argument(
        "--height",
        type=int,
        default=480,
    )

    ap.add_argument(
        "--width",
        type=int,
        default=640,
    )

    ap.add_argument(
        "--num_frames",
        type=int,
        default=17,
    )

    # --------------------------------------------------------------
    # Sampling
    # --------------------------------------------------------------

    ap.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
    )

    ap.add_argument(
        "--sigma_shift",
        type=float,
        default=5.0,
    )

    ap.add_argument(
        "--num_samples",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------

    ap.add_argument(
        "--out_dir",
        required=True,
    )

    # --------------------------------------------------------------
    # Device
    # --------------------------------------------------------------

    ap.add_argument(
        "--device",
        default="cuda",
    )

    ap.add_argument(
        "--dtype",
        default="bf16",
        choices=[
            "bf16",
            "fp32",
        ],
    )

    args = ap.parse_args()

    if args.contact_vae_ckpt or args.force_vae_ckpt:
        raise SystemExit(
            "--contact_vae_ckpt / --force_vae_ckpt were passed, but this is "
            "the video-only ablation arm: it has no physical branches and "
            "writes no gen_contact / gen_force. Passing a VAE here means the "
            "caller expects physical output from a model that cannot produce "
            "it. Use infer_acwm_mot_forcing_multigpu.py for Arm A."
        )

    # --------------------------------------------------------------
    # Device
    # --------------------------------------------------------------

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    device = args.device

    dtype = (
        torch.bfloat16
        if args.dtype == "bf16"
        else torch.float32
    )

    torch.set_grad_enabled(
        False
    )

    print(
        f"[init] device={device} "
        f"dtype={dtype}"
    )

    # --------------------------------------------------------------
    # Model loaded ONCE.
    # --------------------------------------------------------------

    pipe = build_pipe(
        args=args,
        device=device,
        dtype=dtype,
    )

    # --------------------------------------------------------------
    # Dataset
    #
    # IMPORTANT:
    # ACWMDataset performs the same physical/action normalization
    # as training.
    # --------------------------------------------------------------

    ds = ACWMDataset(
        pipeline_json=args.pipeline_json,
        source_root=args.source_root,
        physical_stats_json=args.physical_stats,
        action_stats_json=args.action_stats,
        height=args.height,
        width=args.width,
        repeat=1,
        use_flow=False,
        latent_downsample=16,
    )

    n = min(
        args.num_samples,
        len(ds),
    )

    print(
        f"[dataset] total={len(ds)} "
        f"running={n}"
    )

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Cases
    # --------------------------------------------------------------

    with torch.inference_mode():

        for idx in range(n):

            item = ds[idx]
            raw_sample = ds.samples[idx]

            task = item.get(
                "task",
                raw_sample.get(
                    "task",
                    None,
                ),
            )

            if task is None:
                raise KeyError(
                    f"sample {idx} has no task"
                )

            if task not in SUPPORTED_TASKS:
                raise ValueError(
                    f"sample {idx}: "
                    f"unsupported task={task}"
                )

            out_dir = os.path.join(
                args.out_dir,
                f"sample_{idx:06d}_{task}",
            )

            os.makedirs(
                out_dir,
                exist_ok=True,
            )

            print()
            print("#" * 80)

            print(
                f"[sample {idx}/{n}] "
                f"task={task}"
            )

            print(
                "episode=",
                raw_sample.get(
                    "episode",
                    "",
                ),
            )

            print("#" * 80)

            # ------------------------------------------------------
            # Dataset outputs.
            # ------------------------------------------------------

            observation_image = item[
                "observation_image"
            ]

            actions = item[
                "actions"
            ]

            gt_video = item.get(
                "video",
                None,
            )

            gt_contact = item.get(
                "contact",
                None,
            )

            gt_force = item.get(
                "force",
                None,
            )

            # ------------------------------------------------------
            # Inference.
            # ------------------------------------------------------

            (
                lat_v,
                lat_c,
                lat_f,
            ) = sample(
                pipe=pipe,
                task=task,
                observation_image=observation_image,
                actions=actions,
                gt_video=gt_video,
                gt_contact=gt_contact,
                gt_force=gt_force,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                sigma_shift=args.sigma_shift,
                seed=args.seed + idx,
                device=device,
                dtype=dtype,
            )

            # ------------------------------------------------------
            # Decode.
            # ------------------------------------------------------

            nv = decode_and_save(
                pipe=pipe,
                lat_v=lat_v,
                lat_c=lat_c,
                lat_f=lat_f,
                out_dir=out_dir,
                device=device,
            )

            # ------------------------------------------------------
            # GT copy.
            # ------------------------------------------------------

            copy_gt(
                sample=raw_sample,
                source_root=args.source_root,
                out_dir=out_dir,
            )

            # ------------------------------------------------------
            # Metadata.
            # ------------------------------------------------------

            meta = {
                "sample_idx": idx,
                "task": task,

                "condition_modalities":
                    condition_modalities(task),

                "generated_modalities":
                    generated_modalities(task),

                "episode":
                    raw_sample.get(
                        "episode",
                        "",
                    ),

                "obs_frame_idx":
                    raw_sample.get(
                        "obs_frame_idx",
                    ),

                "observation_frame":
                    raw_sample.get(
                        "observation_frame",
                    ),

                "gen_video_frames": nv,
                "gen_modality_frames": 0,

                "physical_space": (
                    "video-only arm: no contact/force was generated, so "
                    "gen_contact/ and gen_force/ are empty. The gt_contact/ "
                    "and gt_force/ directories still hold raw on-disk values."
                ),
            }

            with open(
                os.path.join(
                    out_dir,
                    "meta.json",
                ),
                "w",
            ) as f:

                json.dump(
                    meta,
                    f,
                    indent=2,
                )

            print(
                f"[saved] {out_dir}"
            )

            del (
                lat_v,
                lat_c,
                lat_f,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print()
    print(
        f"[DONE] processed {n} samples "
        f"-> {args.out_dir}"
    )


if __name__ == "__main__":
    main()