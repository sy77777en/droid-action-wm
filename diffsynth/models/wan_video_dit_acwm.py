# diffsynth/models/wan_video_dit_acwm.py

import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from .wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..core.gradient import gradient_checkpoint_forward


class ACWMWanModel(WanModel):
    """
    ACWM version of Wan DiT.

    First runnable version:
      - video/contact/force latents are passed separately
      - each modality has its own patch embedding
      - tokens are concatenated before the transformer blocks
      - original Wan DiT blocks are reused
      - tokens are split back into video/contact/force before unpatchify

    This is not the final MoT block yet.
    It is the first clean three-modality DiT interface.
    """

    _repeated_blocks = ["DiTBlock"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ACWM uses action context directly.
        # In the first version, do not use CLIP/image cross-attention path.
        if self.has_image_input:
            raise ValueError(
                "ACWMWanModel first version expects has_image_input=False. "
                "Observation image should be fused into latents_video first frame, "
                "not passed through CLIP/image context."
            )

        # Original self.patch_embedding is used for video.
        # Contact / force get their own patch embedding modules.
        # They are initialized from the original video patch embedding.
        self.patch_embedding_contact = copy.deepcopy(self.patch_embedding)
        self.patch_embedding_force = copy.deepcopy(self.patch_embedding)

        # Token-type embedding: video/contact/force.
        # This helps the shared transformer know which modality a token belongs to.
        self.modality_embedding = nn.Parameter(torch.zeros(3, self.dim))
        nn.init.normal_(self.modality_embedding, std=0.02)

    def init_physical_patch_embeddings_from_video(self):
        """
        Initialize contact/force patch embeddings from the loaded video patch embedding.
        Call this after loading pretrained Wan DiT weights.
        """
        self.patch_embedding_contact.load_state_dict(
            self.patch_embedding.state_dict()
        )
        self.patch_embedding_force.load_state_dict(
            self.patch_embedding.state_dict()
        )
    
    # ------------------------------------------------------------------ #
    # Shape utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_latent(name: str, x: torch.Tensor):
        if x is None:
            raise ValueError(f"{name} is required but got None.")

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}.")

        if x.ndim != 5:
            raise ValueError(
                f"{name} must have shape (B, C, T, H, W), got {x.shape}."
            )

    @staticmethod
    def _check_same_shape(
        latents_video: torch.Tensor,
        latents_contact: torch.Tensor,
        latents_force: torch.Tensor,
    ):
        if latents_video.shape != latents_contact.shape:
            raise ValueError(
                "latents_video and latents_contact must have the same shape. "
                f"video={latents_video.shape}, contact={latents_contact.shape}"
            )

        if latents_video.shape != latents_force.shape:
            raise ValueError(
                "latents_video and latents_force must have the same shape. "
                f"video={latents_video.shape}, force={latents_force.shape}"
            )

    # ------------------------------------------------------------------ #
    # ACWM patchify / split / RoPE
    # ------------------------------------------------------------------ #

    def patchify_acwm(
        self,
        latents_video: torch.Tensor,
        latents_contact: torch.Tensor,
        latents_force: torch.Tensor,
    ):
        """
        Patchify three modality latents and concatenate them as one token sequence.

        Args:
            latents_video:   (B, C, T, H, W)
            latents_contact: (B, C, T, H, W)
            latents_force:   (B, C, T, H, W)

        Returns:
            x:
                Concatenated tokens with shape (B, Nv + Nc + Nf, dim).

            grid_size:
                Tuple (f, h, w), the patchified latent grid size.

            token_counts:
                Dict containing token counts for each modality.
        """
        self._check_latent("latents_video", latents_video)
        self._check_latent("latents_contact", latents_contact)
        self._check_latent("latents_force", latents_force)
        self._check_same_shape(latents_video, latents_contact, latents_force)

        # Original video patch embedding.
        x_video = self.patch_embedding(latents_video)

        # Modality-specific patch embeddings.
        x_contact = self.patch_embedding_contact(latents_contact)
        x_force = self.patch_embedding_force(latents_force)

        fv, hv, wv = x_video.shape[2:]
        fc, hc, wc = x_contact.shape[2:]
        ff, hf, wf = x_force.shape[2:]

        if (fv, hv, wv) != (fc, hc, wc):
            raise ValueError(
                "Patchified video/contact grid sizes do not match. "
                f"video={(fv, hv, wv)}, contact={(fc, hc, wc)}"
            )

        if (fv, hv, wv) != (ff, hf, wf):
            raise ValueError(
                "Patchified video/force grid sizes do not match. "
                f"video={(fv, hv, wv)}, force={(ff, hf, wf)}"
            )

        f, h, w = fv, hv, wv

        # (B, dim, f, h, w) -> (B, f*h*w, dim)
        x_video = rearrange(
            x_video,
            "b c f h w -> b (f h w) c",
        ).contiguous()

        x_contact = rearrange(
            x_contact,
            "b c f h w -> b (f h w) c",
        ).contiguous()

        x_force = rearrange(
            x_force,
            "b c f h w -> b (f h w) c",
        ).contiguous()

        # Add modality embedding.
        modality_embedding = self.modality_embedding.to(
            dtype=x_video.dtype,
            device=x_video.device,
        )

        x_video = x_video + modality_embedding[0].view(1, 1, -1)
        x_contact = x_contact + modality_embedding[1].view(1, 1, -1)
        x_force = x_force + modality_embedding[2].view(1, 1, -1)

        token_counts = {
            "video": x_video.shape[1],
            "contact": x_contact.shape[1],
            "force": x_force.shape[1],
        }

        x = torch.cat(
            [x_video, x_contact, x_force],
            dim=1,
        )

        return x, (f, h, w), token_counts

    def build_acwm_freqs(
        self,
        grid_size: Tuple[int, int, int],
        device,
        num_modalities: int = 3,
    ):
        """
        Build RoPE frequencies for concatenated ACWM tokens.

        Original Wan DiT builds RoPE for one video token grid:
            (f*h*w, 1, rope_dim)

        ACWM concatenates video/contact/force tokens:
            (3*f*h*w, 1, rope_dim)

        Therefore we repeat the same spatial-temporal RoPE for each modality.
        """

        f, h, w = grid_size

        base_freqs = torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)

        base_freqs = base_freqs.to(device)

        if num_modalities == 1:
            return base_freqs

        return torch.cat(
            [base_freqs for _ in range(num_modalities)],
            dim=0,
        )

    @staticmethod
    def split_acwm_tokens(
        x: torch.Tensor,
        token_counts: Dict[str, int],
    ):
        """
        Split concatenated ACWM tokens back into video/contact/force tokens.

        Args:
            x: (B, Nv + Nc + Nf, dim)
        """

        nv = token_counts["video"]
        nc = token_counts["contact"]
        nf = token_counts["force"]

        expected = nv + nc + nf
        actual = x.shape[1]

        if actual != expected:
            raise ValueError(
                f"ACWM token length mismatch. Expected {expected}, got {actual}. "
                f"token_counts={token_counts}"
            )

        x_video = x[:, :nv]
        x_contact = x[:, nv:nv + nc]
        x_force = x[:, nv + nc:nv + nc + nf]

        return x_video, x_contact, x_force

    def unpatchify_acwm(
        self,
        x: torch.Tensor,
        grid_size: Tuple[int, int, int],
        token_counts: Dict[str, int],
    ):
        """
        Split ACWM output tokens and unpatchify each modality.
        """

        x_video, x_contact, x_force = self.split_acwm_tokens(
            x,
            token_counts,
        )

        pred_video = self.unpatchify(x_video, grid_size)
        pred_contact = self.unpatchify(x_contact, grid_size)
        pred_force = self.unpatchify(x_force, grid_size)

        return {
            "video": pred_video,
            "contact": pred_contact,
            "force": pred_force,
        }

    # ------------------------------------------------------------------ #
    # ACWM forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        latents_video: torch.Tensor,
        latents_contact: torch.Tensor,
        latents_force: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        """
        ACWM forward.

        Args:
            latents_video/contact/force:
                Noisy latents with shape (B, C, T, H, W).

            timestep:
                Flow matching timestep.

            context:
                Action embedding with shape (B, T_action, dim).
                It should already be projected to self.dim.
                We do NOT call self.text_embedding(context).
        """

        self._check_latent("latents_video", latents_video)
        self._check_latent("latents_contact", latents_contact)
        self._check_latent("latents_force", latents_force)
        self._check_same_shape(latents_video, latents_contact, latents_force)

        if context is None:
            raise ValueError("ACWMWanModel requires action context, got None.")

        if context.ndim != 3:
            raise ValueError(
                "ACWM action context must have shape (B, T_action, dim), "
                f"got {context.shape}."
            )
            
        if context.shape[-1] != self.dim:
            raise ValueError(
                "ACWM action context is expected to be already projected "
                f"to DiT dim={self.dim}, but got last dim={context.shape[-1]}. "
                "Do not pass raw actions here."
            )

        device = latents_video.device
        dtype = latents_video.dtype

        context = context.to(device=device, dtype=dtype)
        timestep = timestep.to(dtype=torch.float32)

        # print("[DIT DEBUG] timestep:", timestep.device, timestep.dtype)
        # print("[DIT DEBUG] time_embedding weight:", self.time_embedding[0].weight.device)
        # emb = sinusoidal_embedding_1d(self.freq_dim, timestep).to(device=device, dtype=dtype)
        # print("[DIT DEBUG] sinusoidal emb:", emb.device, emb.dtype)

        # ------------------------------------------------------------
        # 1. Timestep embedding
        # ------------------------------------------------------------
        # 1. Timestep embedding
        t = self.time_embedding(
            sinusoidal_embedding_1d(
                self.freq_dim,
                timestep,
            ).to(device=device, dtype=dtype)
        )

        t_mod = self.time_projection(t).unflatten(
            1,
            (6, self.dim),
        )

        # ------------------------------------------------------------
        # 2. Patchify three modalities and concatenate tokens
        # ------------------------------------------------------------
        x, grid_size, token_counts = self.patchify_acwm(
            latents_video,
            latents_contact,
            latents_force,
        )

        # ------------------------------------------------------------
        # 3. RoPE for concatenated modality tokens
        # ------------------------------------------------------------
        freqs = self.build_acwm_freqs(
            grid_size,
            device=x.device,
            num_modalities=3,
        )

        if freqs.shape[0] != x.shape[1]:
            raise ValueError(
                "RoPE/token length mismatch. "
                f"freqs length={freqs.shape[0]}, token length={x.shape[1]}"
            )

        # ------------------------------------------------------------
        # 4. Transformer blocks
        # ------------------------------------------------------------
        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x,
                    context,
                    t_mod,
                    freqs,
                )
            else:
                x = block(
                    x,
                    context,
                    t_mod,
                    freqs,
                )

        # ------------------------------------------------------------
        # 5. Shared head, then split/unpatchify
        # ------------------------------------------------------------
        x = self.head(x, t)

        pred = self.unpatchify_acwm(
            x,
            grid_size,
            token_counts,
        )

        return pred
