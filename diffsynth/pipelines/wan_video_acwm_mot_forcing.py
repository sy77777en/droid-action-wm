# diffsynth/pipelines/wan_video_acwm_mot_forcing.py
"""
ACWM Mixture-of-Transformers pipeline.
This reuses EVERYTHING from wan_video_acwm.py (the pipeline units, the noise
initializer, the physical embedders, the action embedder) and only swaps:
  - the DiT model:   ACWMWanModel  ->  ACWMWanModelMoT
  - the conversion:  builds a weight-loaded shared ACWMWanModel first, then
                     derives the MoT model so its per-modality q/k/v/o/ffn
                     branches are deep-copied FROM PRETRAINED WEIGHTS (not from
                     random init).
  - model_fn:        type check accepts ACWMWanModelMoT
Init flow for the DiT:
  original WanModel (pretrained)
    -> convert_wan_dit_to_acwm_dit()      # shared ACWMWanModel, weights loaded
    -> ACWMWanModelMoT(source_blocks=...) # 3x per-modality copies of each block
    -> copy non-block params (patch_embedding*, time_embedding, head,
       modality_embedding, ...) from the shared model so ALL params inherit the
       shared model's (pretrained / initialized) values.
"""

import torch
from typing import Union

from .wan_video import WanVideoUnit_ShapeChecker, ModelConfig
from .wan_video_acwm import (
    ACWMWanVideoPipeline,
    convert_wan_dit_to_acwm_dit,
    ACWMNoiseInitializer,
    ACWMInputVideoPhysicalEmbedder,
    ACWMObservationImageEmbedder,
    ACWMActionEmbedder,
)

from ..core.device.npu_compatible_device import get_device_type
from ..models.wan_video_dit import WanModel
from ..models.wan_video_dit_acwm import ACWMWanModel
from ..models.wan_video_dit_acwm_mot_forcing import ACWMWanModelMoTForcing
from ..models.wan_video_text_encoder import HuggingfaceTokenizer
from transformers import Wav2Vec2Processor


def convert_wan_dit_to_acwm_mot_dit(dit: WanModel):
    """
    Convert an original WanModel into ACWMWanModelMoT.
    Step 1: build a weight-loaded shared ACWMWanModel (reuses the existing,
            already-tested conversion). Its blocks hold pretrained weights and
            its physical patch embeddings are initialized from the video one.
    Step 2: build ACWMWanModelMoT, deep-copying each MoT block FROM the shared
            model's pretrained blocks (source_blocks=shared.blocks).
    Step 3: copy all NON-block parameters (patch_embedding, patch_embedding_
            contact/force, modality_embedding, time_embedding, time_projection,
            head, text_embedding, ...) from the shared model, so every MoT
            parameter inherits the shared model's values. Block params are left
            as the per-modality deep-copies from step 2.
    """
    if isinstance(dit, ACWMWanModelMoTForcing):
        return dit

    # Step 1: weight-loaded shared ACWMWanModel (existing tested path)
    shared = convert_wan_dit_to_acwm_dit(dit)
    if not isinstance(shared, ACWMWanModel):
        raise TypeError(
            f"convert_wan_dit_to_acwm_dit returned {type(shared)}, expected ACWMWanModel.")

    param = next(shared.parameters())
    device, dtype = param.device, param.dtype

    # Rebuild the config the same way the shared model was configured.
    from .wan_video_acwm import _infer_acwm_dit_config_from_wan_dit
    config = _infer_acwm_dit_config_from_wan_dit(dit)

    # Step 2: MoT model, blocks deep-copied from the shared model's pretrained blocks
    mot = ACWMWanModelMoTForcing(source_blocks=shared.blocks, **config)

    # Step 3: copy all non-block params from the shared model.
    # MoT block param names (self_attn_video.* etc.) differ from shared block
    # names (self_attn.*), so load_state_dict(strict=False) will:
    #   - load every non-block key (patch_embedding, time_embedding, head,
    #     modality_embedding, patch_embedding_contact/force, ...) that matches
    #   - skip the shared "blocks.*" keys (they don't exist in MoT) -> that's
    #     fine, MoT blocks already have pretrained weights from step 2.
    shared_sd = shared.state_dict()
    non_block_sd = {k: v for k, v in shared_sd.items() if not k.startswith("blocks.")}
    load_result = mot.load_state_dict(non_block_sd, strict=False)

    # sanity: the only missing keys should be MoT block params; the only
    # unexpected keys should be none (we filtered blocks out).
    missing_non_block = [k for k in load_result.missing_keys if not k.startswith("blocks.")]
    if missing_non_block:
        print(f"[ACWM-MoT] warning: non-block params not loaded: {missing_non_block[:8]}"
              f"{' ...' if len(missing_non_block) > 8 else ''}")
    if load_result.unexpected_keys:
        print(f"[ACWM-MoT] warning: unexpected keys: {load_result.unexpected_keys[:8]}"
              f"{' ...' if len(load_result.unexpected_keys) > 8 else ''}")

    mot = mot.to(device=device, dtype=dtype)
    return mot


class ACWMWanVideoPipelineMoTForcing(ACWMWanVideoPipeline):
    """Same as ACWMWanVideoPipeline but the DiT is ACWMWanModelMoT.
    Units and loss are identical (reused from wan_video_acwm)."""

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        # units/model_fn/in_iteration_models are inherited unchanged; just swap
        # the model_fn to the MoT-aware one (type check differs).
        self.model_fn = model_fn_wan_acwm_mot_forcing

    @classmethod
    def from_pretrained(
        cls,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list = [],
        tokenizer_config=None,
        audio_processor_config=None,
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        # Reuse the parent loader but override DiT conversion to MoT.
        # Simplest correct approach: call parent from_pretrained, then rebuild
        # dit as MoT from the (already converted) shared ACWM dit is NOT ideal
        # (parent already converted to shared ACWM). Instead we replicate the
        # loader minimally here.
        #
        # To avoid duplicating the entire parent loader, we call the parent to
        # get a fully-built pipeline whose .dit is a shared ACWMWanModel, then
        # convert that shared model in-place to MoT.
        pipe = super().from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
            redirect_common_files=redirect_common_files,
            use_usp=use_usp,
            vram_limit=vram_limit,
        )
        # pipe.dit is a shared ACWMWanModel (already weight-loaded). Convert to
        # MoT by deep-copying its pretrained blocks into per-modality branches.
        pipe.dit = _shared_acwm_to_mot(pipe.dit)
        if getattr(pipe, "dit2", None) is not None:
            pipe.dit2 = _shared_acwm_to_mot(pipe.dit2)
        pipe.model_fn = model_fn_wan_acwm_mot_forcing
        return pipe


def _shared_acwm_to_mot(shared: ACWMWanModel) -> ACWMWanModelMoTForcing:
    """Derive a MoT model from an ALREADY weight-loaded shared ACWMWanModel."""
    if isinstance(shared, ACWMWanModelMoTForcing):
        return shared
    param = next(shared.parameters())
    device, dtype = param.device, param.dtype

    from .wan_video_acwm import _infer_acwm_dit_config_from_wan_dit
    # Reconstruct config from the shared model itself (it IS a WanModel).
    config = _infer_acwm_dit_config_from_wan_dit(shared)

    mot = ACWMWanModelMoTForcing(
        source_blocks=shared.blocks,
        **config,
    )
    shared_sd = shared.state_dict()
    non_block_sd = {k: v for k, v in shared_sd.items() if not k.startswith("blocks.")}
    load_result = mot.load_state_dict(non_block_sd, strict=False)
    missing_non_block = [k for k in load_result.missing_keys if not k.startswith("blocks.")]
    if missing_non_block:
        print(f"[ACWM-MoT] warning: non-block params not loaded: {missing_non_block[:8]}")
    if load_result.unexpected_keys:
        print(f"[ACWM-MoT] warning: unexpected keys: {load_result.unexpected_keys[:8]}")
    mot = mot.to(device=device, dtype=dtype)
    return mot


def model_fn_wan_acwm_mot_forcing(
    dit: ACWMWanModelMoTForcing,
    latents_video: torch.Tensor = None,
    latents_contact: torch.Tensor = None,
    latents_force: torch.Tensor = None,
    timestep_video: torch.Tensor = None,
    timestep_contact: torch.Tensor = None,
    timestep_force: torch.Tensor = None,
    context: torch.Tensor = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    if not isinstance(dit, ACWMWanModelMoTForcing):
        raise TypeError(
            f"model_fn_wan_acwm_mot_forcing expects "
            f"ACWMWanModelMoTForcing, got {type(dit)}."
        )

    return dit(
        latents_video=latents_video,
        latents_contact=latents_contact,
        latents_force=latents_force,
        timestep_video=timestep_video,
        timestep_contact=timestep_contact,
        timestep_force=timestep_force,
        context=context,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        **kwargs,
    )