import copy
import math
import torch
from typing import Union

from .wan_video import (
    WanVideoPipeline,
    WanVideoUnit_ShapeChecker,
    ModelConfig,
)

from transformers import Wav2Vec2Processor

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion.base_pipeline import PipelineUnit
from ..models.wan_video_text_encoder import HuggingfaceTokenizer
from ..models.wan_video_dit import WanModel
from ..models.wan_video_dit_acwm import ACWMWanModel


def _infer_acwm_dit_config_from_wan_dit(dit: WanModel):
    """
    Infer ACWMWanModel constructor config from a loaded original WanModel.
    This lets us reuse the original Wan checkpoint but instantiate ACWMWanModel.
    """

    patch_size = dit.patch_size
    out_dim = dit.head.head.out_features // math.prod(patch_size)

    return dict(
        dim=dit.dim,
        in_dim=dit.in_dim,
        ffn_dim=dit.blocks[0].ffn_dim,
        out_dim=out_dim,
        text_dim=dit.text_embedding[0].in_features,
        freq_dim=dit.freq_dim,
        eps=dit.blocks[0].norm1.eps,
        patch_size=patch_size,
        num_heads=dit.blocks[0].num_heads,
        num_layers=len(dit.blocks),
        has_image_input=dit.has_image_input,
        has_image_pos_emb=getattr(dit, "has_image_pos_emb", False),
        has_ref_conv=getattr(dit, "has_ref_conv", False),
        add_control_adapter=getattr(dit, "control_adapter", None) is not None,
        seperated_timestep=getattr(dit, "seperated_timestep", False),
        require_vae_embedding=getattr(dit, "require_vae_embedding", True),
        require_clip_embedding=getattr(dit, "require_clip_embedding", True),
        fuse_vae_embedding_in_latents=getattr(
            dit,
            "fuse_vae_embedding_in_latents",
            False,
        ),
        wantodance_enable_music_inject=getattr(
            dit,
            "wantodance_enable_music_inject",
            False,
        ),
        wantodance_enable_refimage=getattr(
            dit,
            "wantodance_enable_refimage",
            False,
        ),
        wantodance_enable_refface=getattr(
            dit,
            "wantodance_enable_refface",
            False,
        ),
        wantodance_enable_global=getattr(
            dit,
            "wantodance_enable_global",
            False,
        ),
        wantodance_enable_dynamicfps=getattr(
            dit,
            "wantodance_enable_dynamicfps",
            False,
        ),
        wantodance_enable_unimodel=getattr(
            dit,
            "wantodance_enable_unimodel",
            False,
        ),
    )


def convert_wan_dit_to_acwm_dit(dit: WanModel):
    """
    Convert a loaded original WanModel into ACWMWanModel.

    Important:
        Original Wan checkpoint does not contain:
            patch_embedding_contact
            patch_embedding_force
            modality_embedding

        So we load original weights with strict=False, then initialize
        contact/force patch embeddings from loaded video patch embedding.
    """

    if isinstance(dit, ACWMWanModel):
        return dit

    if not isinstance(dit, WanModel):
        raise TypeError(
            f"Expected WanModel or ACWMWanModel, got {type(dit)}."
        )

    param = next(dit.parameters())
    device = param.device
    dtype = param.dtype

    config = _infer_acwm_dit_config_from_wan_dit(dit)

    acwm_dit = ACWMWanModel(**config)

    load_result = acwm_dit.load_state_dict(
        dit.state_dict(),
        strict=False,
    )

    # These missing keys are expected because ACWM adds new modules.
    expected_missing_prefixes = (
        "patch_embedding_contact.",
        "patch_embedding_force.",
        "modality_embedding",
    )

    unexpected_missing = [
        key for key in load_result.missing_keys
        if not key.startswith(expected_missing_prefixes)
    ]

    if len(unexpected_missing) > 0:
        print(
            "[ACWM] Warning: unexpected missing keys when loading Wan weights "
            f"into ACWMWanModel: {unexpected_missing}"
        )

    if len(load_result.unexpected_keys) > 0:
        print(
            "[ACWM] Warning: unexpected keys when loading Wan weights "
            f"into ACWMWanModel: {load_result.unexpected_keys}"
        )

    # Important: call this after loading pretrained weights.
    if not hasattr(acwm_dit, "init_physical_patch_embeddings_from_video"):
        raise AttributeError(
            "ACWMWanModel must define `init_physical_patch_embeddings_from_video()`. "
            "This is needed to initialize contact/force patch embeddings from the "
            "loaded video patch embedding."
        )
    
    acwm_dit.init_physical_patch_embeddings_from_video()
    acwm_dit = acwm_dit.to(device=device, dtype=dtype)

    return acwm_dit


class ACWMWanVideoPipeline(WanVideoPipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)

        # Extra encoders for your task
        self.action_encoder = None
        self.action_proj = None
        self.contact_vae = None
        self.force_vae = None

        self.units = [
            WanVideoUnit_ShapeChecker(),
            ACWMNoiseInitializer(),
            ACWMInputVideoPhysicalEmbedder(),
            ACWMObservationImageEmbedder(),
            ACWMActionEmbedder(),
        ]

        self.post_units = []
        self.model_fn = model_fn_wan_acwm
        # ACWM loss should only pass the DiT into model_fn.
        # Do not pass motion_controller / vace / vap / animate_adapter.
        self.in_iteration_models = ("dit",)
        self.compilable_models = ["dit"]

    @classmethod
    def from_pretrained(
        cls,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        # Redirect common files, same as original WanVideoPipeline.
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "models_t5_umt5-xxl-enc-bf16.safetensors",
                ),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
                ),
                "Wan2.1_VAE.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "Wan2.1_VAE.safetensors",
                ),
                "Wan2.2_VAE.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "Wan2.2_VAE.safetensors",
                ),
            }

            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if (
                    model_config.origin_file_pattern in redirect_dict
                    and model_config.model_id
                    != redirect_dict[model_config.origin_file_pattern][0]
                ):
                    print(
                        f"To avoid repeatedly downloading model files, "
                        f"({model_config.model_id}, {model_config.origin_file_pattern}) "
                        f"is redirected to {redirect_dict[model_config.origin_file_pattern]}."
                    )
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[
                        model_config.origin_file_pattern
                    ][1]

        if use_usp:
            from ..utils.xfuser import initialize_usp
            initialize_usp(device)
            import torch.distributed as dist
            from ..core.device.npu_compatible_device import get_device_name

            if dist.is_available() and dist.is_initialized():
                device = get_device_name()

        # IMPORTANT:
        # Use cls instead of WanVideoPipeline.
        pipe = cls(device=device, torch_dtype=torch_dtype)

        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Fetch models.
        # You mainly need dit + vae for ACWM.
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")

        # dit = model_pool.fetch_model("wan_video_dit", index=2)
        # if isinstance(dit, list):
        #     pipe.dit, pipe.dit2 = dit
        # else:
        #     pipe.dit = dit

        dit = model_pool.fetch_model("wan_video_dit", index=2)

        if isinstance(dit, list):
            pipe.dit = convert_wan_dit_to_acwm_dit(dit[0])
            pipe.dit2 = convert_wan_dit_to_acwm_dit(dit[1])
        else:
            pipe.dit = convert_wan_dit_to_acwm_dit(dit)

        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")

        # These are not required for your minimal ACWM pipeline,
        # but keeping them does not hurt if the model pool has them.
        pipe.motion_controller = model_pool.fetch_model("wan_video_motion_controller")

        vace = model_pool.fetch_model("wan_video_vace", index=2)
        if isinstance(vace, list):
            pipe.vace, pipe.vace2 = vace
        else:
            pipe.vace = vace

        pipe.vap = model_pool.fetch_model("wan_video_vap")
        pipe.audio_encoder = model_pool.fetch_model("wans2v_audio_encoder")
        pipe.animate_adapter = model_pool.fetch_model("wan_video_animate_adapter")

        # Size division factor.
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Tokenizer is not necessary if you fully replace text with actions,
        # but keeping it makes the pipeline compatible with original loading.
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(
                name=tokenizer_config.path,
                seq_len=512,
                clean="whitespace",
            )

        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary()
            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(
                audio_processor_config.path
            )

        if use_usp:
            pipe.enable_usp()

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


class ACWMNoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise_video", "noise_contact", "noise_force"),
        )

    def process(self, pipe, height, width, num_frames, seed=None, rand_device="cpu"):
        length = (num_frames - 1) // 4 + 1
        h = height // pipe.vae.upsampling_factor
        w = width // pipe.vae.upsampling_factor
        c = pipe.vae.model.z_dim

        shape = (1, c, length, h, w)

        noise_video = pipe.generate_noise (shape, seed=seed, rand_device=rand_device)
        noise_contact = pipe.generate_noise(shape, seed=seed + 1 if seed is not None else None, rand_device=rand_device)
        noise_force = pipe.generate_noise(shape, seed=seed + 2 if seed is not None else None, rand_device=rand_device)

        return {
            "noise_video": noise_video,
            "noise_contact": noise_contact,
            "noise_force": noise_force,
        }


class ACWMInputVideoPhysicalEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "video",
                "contact",
                "force",
                "noise_video",
                "noise_contact",
                "noise_force",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=(
                "latents_video",
                "latents_contact",
                "latents_force",
                "input_latents_video",
                "input_latents_contact",
                "input_latents_force",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        video,
        contact,
        force,
        noise_video,
        noise_contact,
        noise_force,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        pipe.load_models_to_device(self.onload_model_names)

        # RGB video latent
        video_tensor = pipe.preprocess_video(video)
        input_latents_video = pipe.vae.encode(
            video_tensor,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)

        # Contact latent: contact should be Tensor (B, 2, T, H, W)
        if pipe.contact_vae is not None and contact is not None:
            contact = contact.to(device=pipe.device, dtype=pipe.torch_dtype)
            input_latents_contact = pipe.contact_vae.encode_for_dit(
                contact,
                sample=False,
                apply_scale=True,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
        else:
            input_latents_contact = torch.zeros_like(input_latents_video)

        # Force latent: force should be Tensor (B, 6, T, H, W)
        if pipe.force_vae is not None and force is not None:
            force = force.to(device=pipe.device, dtype=pipe.torch_dtype)
            input_latents_force = pipe.force_vae.encode_for_dit(
                force,
                sample=False,
                apply_scale=True,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
        else:
            input_latents_force = torch.zeros_like(input_latents_video)

        return {
            # These are noisy states used by inference or loss.
            # In training loss we will add noise to input_latents_* again.
            # For now initialize them as pure noise.
            "latents_video": noise_video,
            "latents_contact": noise_contact,
            "latents_force": noise_force,

            # These are clean target latents.
            "input_latents_video": input_latents_video,
            "input_latents_contact": input_latents_contact,
            "input_latents_force": input_latents_force,
        }


class ACWMObservationImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "observation_image",
                "latents_video",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=(
                "latents_video",
                "first_frame_latents_video",
                "first_frame_latents",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        observation_image,
        latents_video,
        height,
        width,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        if observation_image is None:
            raise ValueError(
                "ACWM requires `observation_image`. "
                "It will be VAE-encoded as `first_frame_latents_video`."
            )

        pipe.load_models_to_device(self.onload_model_names)

        image = pipe.preprocess_image(
            observation_image.resize((width, height))
        ).transpose(0, 1)

        z = pipe.vae.encode(
            [image],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)

        latents_video[:, :, 0:1] = z

        return {
            "latents_video": latents_video,
            "first_frame_latents_video": z,
        
            # Backward compatibility with original TI2V naming.
            "first_frame_latents": z,
        }


class ACWMActionEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("actions", "drop_action"),
            output_params=("context",),
        )

    def process(self, pipe, actions, drop_action=False):
        if actions is None:
            raise ValueError("ACWMActionEmbedder requires `actions`.")

        if pipe.action_encoder is None:
            raise ValueError("pipe.action_encoder is not initialized.")

        actions = actions.to(device=pipe.device, dtype=pipe.torch_dtype)

        # `drop_action` arrives as None from callers that never set it. Only
        # pass the kwarg when it is actually requested, so action encoders that
        # predate it (WanActionEncoder, FakeActionEncoder) keep working.
        if drop_action:
            action_context = pipe.action_encoder(actions, drop_action=True)
        else:
            action_context = pipe.action_encoder(actions)

        # Optional: projection to dit.dim, if action_encoder does not already do it.
        if pipe.action_proj is not None:
            action_context = pipe.action_proj(action_context)

        return {
            "context": action_context.to(
                dtype=pipe.torch_dtype,
                device=pipe.device,
            )
        }


def model_fn_wan_acwm(
    dit: ACWMWanModel,
    latents_video: torch.Tensor = None,
    latents_contact: torch.Tensor = None,
    latents_force: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    """
    ACWM model function.

    This directly calls ACWMWanModel.forward(), which returns:
        {
            "video": pred_video,
            "contact": pred_contact,
            "force": pred_force,
        }
    """

    if not isinstance(dit, ACWMWanModel):
        raise TypeError(
            "model_fn_wan_acwm expects `dit` to be ACWMWanModel, "
            f"but got {type(dit)}. "
            "Check whether from_pretrained() converted the original WanModel."
        )

    return dit(
        latents_video=latents_video,
        latents_contact=latents_contact,
        latents_force=latents_force,
        timestep=timestep,
        context=context,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        **kwargs,
    )
        