"""
ACWM Mixture-of-Transformers (MoT) DiT with modality forcing.

Granularity:
  - per-modality q/k/v/o projections
  - per-modality FFN
  - per-modality modulation base
  - per-modality diffusion timestep conditioning

Self-attention computation is still full-sequence:
video/contact/force tokens attend to each other after their modality-specific
projections are concatenated.

For modality forcing, each modality can be at a different diffusion timestep:

    condition modality -> clean latent -> timestep = 0
    target modality    -> noisy latent -> timestep = sampled t

The three modalities share the SAME time_embedding and time_projection weights,
but each modality evaluates them with its own timestep.

Example:

    O + A + video + contact -> force

becomes:

    video:
        clean latent
        timestep_video = 0

    contact:
        clean latent
        timestep_contact = 0

    force:
        noisy latent
        timestep_force = t

Everything else:
  - patchify
  - token concatenation
  - cross-modal full-sequence self-attention
  - RoPE
  - framewise action cross-attention
  - unpatchify

is unchanged from the original ACWM MoT implementation.
"""

import copy

import torch
import torch.nn as nn
from einops import rearrange

from .wan_video_dit import (
    sinusoidal_embedding_1d,
    modulate,
    rope_apply,
    flash_attention,
    DiTBlock,
)
from .wan_video_dit_acwm import ACWMWanModel
from ..core.gradient import gradient_checkpoint_forward


MODALITIES = ("video",)


def _split3(x, token_counts):
    nv = token_counts["video"]
    nc = token_counts["contact"]
    nf = token_counts["force"]

    return (
        x[:, :nv],
        x[:, nv:nv + nc],
        x[:, nv + nc:nv + nc + nf],
    )


def _split_freqs3(freqs, token_counts):
    """
    freqs:
        (Nv + Nc + Nf, 1, rope_dim)

    Split using the same token boundaries as the concatenated latent sequence.
    """
    nv = token_counts["video"]
    nc = token_counts["contact"]
    nf = token_counts["force"]

    return (
        freqs[:nv],
        freqs[nv:nv + nc],
        freqs[nv + nc:nv + nc + nf],
    )


class MoTDiTBlock(nn.Module):
    """
    MoT block with:

      - per-modality self-attention q/k/v/o
      - per-modality FFN
      - per-modality modulation base
      - per-modality timestep modulation
      - shared full-sequence self-attention computation
      - shared action k/v with per-modality action q/o

    All modality-specific branches are initialized by deep-copying an existing
    shared DiTBlock, preserving the pretrained initialization.
    """

    def __init__(self, base_block: DiTBlock):
        super().__init__()

        self.dim = base_block.dim
        self.num_heads = base_block.num_heads
        self.ffn_dim = base_block.ffn_dim

        # ------------------------------------------------------------
        # Per-modality self-attention / FFN / modulation
        # ------------------------------------------------------------
        for m in MODALITIES:
            setattr(
                self,
                f"self_attn_{m}",
                copy.deepcopy(base_block.self_attn),
            )

            setattr(
                self,
                f"ffn_{m}",
                copy.deepcopy(base_block.ffn),
            )

            setattr(
                self,
                f"modulation_{m}",
                nn.Parameter(
                    base_block.modulation.detach().clone()
                ),
            )

        # ------------------------------------------------------------
        # Action cross-attention
        #
        # Action is not itself a mixture modality.
        #
        # k/v:
        #   shared across video/contact/force
        #
        # q/o:
        #   modality specific
        # ------------------------------------------------------------
        self.cross_attn = copy.deepcopy(base_block.cross_attn)

        for m in MODALITIES:
            setattr(
                self,
                f"cross_q_{m}",
                copy.deepcopy(base_block.cross_attn.q),
            )

            setattr(
                self,
                f"cross_o_{m}",
                copy.deepcopy(base_block.cross_attn.o),
            )

            setattr(
                self,
                f"cross_norm_q_{m}",
                copy.deepcopy(base_block.cross_attn.norm_q),
            )

        # ------------------------------------------------------------
        # Norms / gate
        # ------------------------------------------------------------
        self.norm1 = copy.deepcopy(base_block.norm1)
        self.norm2 = copy.deepcopy(base_block.norm2)
        self.norm3 = copy.deepcopy(base_block.norm3)
        self.gate = copy.deepcopy(base_block.gate)

    # ------------------------------------------------------------------
    # Self-attention helper
    # ------------------------------------------------------------------
    def _sa_project(
        self,
        sa,
        x,
        freqs,
    ):
        """
        Project q/k/v for one modality and apply RoPE to q/k.
        """
        q = sa.norm_q(sa.q(x))
        k = sa.norm_k(sa.k(x))
        v = sa.v(x)

        q = rope_apply(
            q,
            freqs,
            sa.num_heads,
        )

        k = rope_apply(
            k,
            freqs,
            sa.num_heads,
        )

        return q, k, v

    # ------------------------------------------------------------------
    # Action grouping
    # ------------------------------------------------------------------
    def _group_actions(
        self,
        context: torch.Tensor,
        actions_per_latent: int = 4,
    ) -> torch.Tensor:
        """
        Group sequential action tokens using a fixed 1:4 latent/action mapping.

        Args:
            context:
                (B, T_action, D)

            actions_per_latent:
                Number of action tokens assigned to one latent frame.

        Returns:
            (B,
             T_action // actions_per_latent,
             actions_per_latent,
             D)
        """

        if context.ndim != 3:
            raise ValueError(
                f"Expected context shape (B, T_action, D), "
                f"got {context.shape}."
            )

        B, seq_action, D = context.shape

        if seq_action % actions_per_latent != 0:
            raise ValueError(
                f"Action sequence length {seq_action} is not divisible by "
                f"actions_per_latent={actions_per_latent}."
            )

        num_action_latents = (
            seq_action // actions_per_latent
        )

        return context.reshape(
            B,
            num_action_latents,
            actions_per_latent,
            D,
        )

    # ------------------------------------------------------------------
    # Framewise action cross-attention
    # ------------------------------------------------------------------
    def _framewise_action_cross_attention(
        self,
        x: torch.Tensor,
        action_context: torch.Tensor,
        grid_size,
        modality: str,
        actions_per_latent: int = 4,
    ) -> torch.Tensor:
        """
        The final seq_action // actions_per_latent latent frames attend to
        corresponding action groups.

        Earlier observation/history latent frames receive zero action
        cross-attention output.
        """

        f, h, w = grid_size

        B, N, D = x.shape

        spatial_tokens = h * w

        if N != f * spatial_tokens:
            raise ValueError(
                f"{modality}: N={N}, "
                f"expected {f * spatial_tokens}, "
                f"grid_size={grid_size}."
            )

        action_groups = self._group_actions(
            action_context,
            actions_per_latent=actions_per_latent,
        )

        num_action_latents = action_groups.shape[1]

        if num_action_latents > f:
            raise ValueError(
                f"Action groups require {num_action_latents} latent frames, "
                f"but latent sequence only has {f} frames."
            )

        # Number of observation/history latent frames.
        prefix_frames = f - num_action_latents

        if not hasattr(self, "_framewise_verified"):
            print(
                "[Framewise verified]",
                f"grid_size={(f, h, w)}",
                f"x={tuple(x.shape)}",
                f"context={tuple(action_context.shape)}",
                f"action_groups={tuple(action_groups.shape)}",
                f"prefix_frames={prefix_frames}",
                f"modality={modality}",
            )

            self._framewise_verified = True

        # ------------------------------------------------------------
        # (B, f*h*w, D)
        # ->
        # (B, f, h*w, D)
        # ------------------------------------------------------------
        x_framewise = x.reshape(
            B,
            f,
            spatial_tokens,
            D,
        )

        # Only future/action-aligned latent frames receive action conditioning.
        x_action = x_framewise[:, prefix_frames:]

        q_proj = getattr(
            self,
            f"cross_q_{modality}",
        )

        q_norm = getattr(
            self,
            f"cross_norm_q_{modality}",
        )

        out_proj = getattr(
            self,
            f"cross_o_{modality}",
        )

        q = q_norm(
            q_proj(
                self.norm3(x_action)
            )
        )

        ca = self.cross_attn

        k = ca.norm_k(
            ca.k(action_groups)
        )

        v = ca.v(
            action_groups
        )

        # Independent attention for every action-aligned latent frame.
        q = q.reshape(
            B * num_action_latents,
            spatial_tokens,
            D,
        )

        k = k.reshape(
            B * num_action_latents,
            actions_per_latent,
            D,
        )

        v = v.reshape(
            B * num_action_latents,
            actions_per_latent,
            D,
        )

        action_out = ca.attn(
            q,
            k,
            v,
        )

        action_out = out_proj(
            action_out
        )

        action_out = action_out.reshape(
            B,
            num_action_latents,
            spatial_tokens,
            D,
        )

        # Observation/history frames receive no future-action attention.
        prefix_out = torch.zeros(
            B,
            prefix_frames,
            spatial_tokens,
            D,
            dtype=action_out.dtype,
            device=action_out.device,
        )

        out = torch.cat(
            [
                prefix_out,
                action_out,
            ],
            dim=1,
        )

        return out.reshape(
            B,
            N,
            D,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x,
        context,
        t_mod_video,
        t_mod_contact,
        t_mod_force,
        freqs,
        token_counts,
        grid_size,
    ):
        """
        Each modality receives its own timestep modulation.

        The underlying modulation parameters remain modality-specific as before,
        but now the time-dependent component is also modality-specific.
        """

        # ------------------------------------------------------------
        # Per-modality timestep modulation
        # ------------------------------------------------------------
        t_mod_by_m = {
            "video": t_mod_video,
        }

        # ------------------------------------------------------------
        # Split token sequence and RoPE frequencies
        # ------------------------------------------------------------
        # Single modality: the whole token sequence is video, so there is
        # nothing to split -- _split3/_split_freqs3 are 3-modality helpers.
        branches = {
            "video": (
                x,
                freqs,
            ),
        }

        # ------------------------------------------------------------
        # Build modulation values independently for each modality
        # ------------------------------------------------------------
        mod = {}

        for m in MODALITIES:
            t_mod_m = t_mod_by_m[m]

            has_seq = len(t_mod_m.shape) == 4
            chunk_dim = 2 if has_seq else 1

            base = getattr(
                self,
                f"modulation_{m}",
            ).to(
                dtype=t_mod_m.dtype,
                device=t_mod_m.device,
            )

            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = (
                base + t_mod_m
            ).chunk(
                6,
                dim=chunk_dim,
            )

            if has_seq:
                (
                    shift_msa,
                    scale_msa,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = (
                    shift_msa.squeeze(2),
                    scale_msa.squeeze(2),
                    gate_msa.squeeze(2),
                    shift_mlp.squeeze(2),
                    scale_mlp.squeeze(2),
                    gate_mlp.squeeze(2),
                )

            mod[m] = (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            )

        # ------------------------------------------------------------
        # Self-attention
        #
        # Per-modality q/k/v projection
        # ->
        # concatenate
        # ->
        # full-sequence attention
        # ->
        # split
        # ->
        # per-modality output projection
        # ------------------------------------------------------------
        q_list = []
        k_list = []
        v_list = []

        for m in MODALITIES:
            xm, fqm = branches[m]

            shift_msa = mod[m][0]
            scale_msa = mod[m][1]

            xin = modulate(
                self.norm1(xm),
                shift_msa,
                scale_msa,
            )

            sa = getattr(
                self,
                f"self_attn_{m}",
            )

            q, k, v = self._sa_project(
                sa,
                xin,
                fqm,
            )

            q_list.append(q)
            k_list.append(k)
            v_list.append(v)

        Q = torch.cat(
            q_list,
            dim=1,
        )

        K = torch.cat(
            k_list,
            dim=1,
        )

        V = torch.cat(
            v_list,
            dim=1,
        )

        # Full cross-modal sequence attention.
        attn_out = flash_attention(
            Q,
            K,
            V,
            num_heads=self.num_heads,
        )

        # Single modality: the attention output is entirely video.
        attn_by_m = {
            "video": attn_out,
        }

        new_x = {}

        for m in MODALITIES:
            xm, _ = branches[m]

            sa = getattr(
                self,
                f"self_attn_{m}",
            )

            gate_msa = mod[m][2]

            o = sa.o(
                attn_by_m[m]
            )

            new_x[m] = self.gate(
                xm,
                gate_msa,
                o,
            )

        # ------------------------------------------------------------
        # Framewise action cross-attention
        # ------------------------------------------------------------
        for m in MODALITIES:
            framewise_out = (
                self._framewise_action_cross_attention(
                    x=new_x[m],
                    action_context=context,
                    grid_size=grid_size,
                    modality=m,
                )
            )

            new_x[m] = (
                new_x[m]
                + framewise_out
            )

        # ------------------------------------------------------------
        # Per-modality FFN
        # ------------------------------------------------------------
        for m in MODALITIES:
            xm = new_x[m]

            shift_mlp = mod[m][3]
            scale_mlp = mod[m][4]
            gate_mlp = mod[m][5]

            ffn = getattr(
                self,
                f"ffn_{m}",
            )

            xin = modulate(
                self.norm2(xm),
                shift_mlp,
                scale_mlp,
            )

            xm = self.gate(
                xm,
                gate_mlp,
                ffn(xin),
            )

            new_x[m] = xm

        # ------------------------------------------------------------
        # Concatenate back in canonical modality order
        # ------------------------------------------------------------
        return torch.cat(
            [
                new_x["video"],
            ],
            dim=1,
        )


class ACWMWanModelVideoOnly(ACWMWanModel):
    """
    ACWM DiT with Mixture-of-Transformers blocks and per-modality diffusion
    timestep conditioning.

    The three modalities share the same time embedding network weights but can
    receive different timestep values.
    """

    _repeated_blocks = ["MoTDiTBlock"]

    def __init__(
        self,
        *args,
        source_blocks=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        # ------------------------------------------------------------
        # Initialize MoT blocks from pretrained shared blocks when provided.
        # ------------------------------------------------------------
        src = (
            source_blocks
            if source_blocks is not None
            else self.blocks
        )

        if len(src) != len(self.blocks):
            raise ValueError(
                f"source_blocks length {len(src)} "
                f"!= num_layers {len(self.blocks)}."
            )

        mot_blocks = nn.ModuleList()

        for block in src:
            mot_blocks.append(
                MoTDiTBlock(block)
            )

        self.blocks = mot_blocks

    def init_mot_from_shared(self):
        """
        No-op placeholder.

        MoT blocks are initialized from the shared blocks during construction.
        """
        pass

    # ------------------------------------------------------------------
    # Time embedding helper
    # ------------------------------------------------------------------
    def _build_time_condition(
        self,
        timestep: torch.Tensor,
        device,
        dtype,
    ):
        """
        Apply the SAME shared time_embedding/time_projection modules to one
        modality's timestep.

        Returns:
            t:
                time embedding used by the final head

            t_mod:
                projected modulation embedding used by DiT blocks
        """

        timestep = timestep.to(
            device=device,
            dtype=torch.float32,
        )

        t = self.time_embedding(
            sinusoidal_embedding_1d(
                self.freq_dim,
                timestep,
            ).to(
                device=device,
                dtype=dtype,
            )
        )

        t_mod = self.time_projection(
            t
        ).unflatten(
            1,
            (
                6,
                self.dim,
            ),
        )

        return t, t_mod

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def patchify_video_only(self, latents_video: torch.Tensor):
        """Single-modality counterpart of ACWMWanModel.patchify_acwm.

        Identical to the video third of the parent: same patch_embedding, same
        rearrange, same modality_embedding[0] offset -- so a video token here is
        byte-for-byte what the 3-modality model would produce for video. Only
        the contact/force branches and their concatenation are absent.
        """
        self._check_latent("latents_video", latents_video)

        x_video = self.patch_embedding(latents_video)
        f, h, w = x_video.shape[2:]

        x_video = rearrange(
            x_video,
            "b c f h w -> b (f h w) c",
        ).contiguous()

        modality_embedding = self.modality_embedding.to(
            dtype=x_video.dtype,
            device=x_video.device,
        )
        x_video = x_video + modality_embedding[0].view(1, 1, -1)

        token_counts = {"video": x_video.shape[1]}

        return x_video, (f, h, w), token_counts

    def forward(
        self,
        latents_video: torch.Tensor,
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
        # ------------------------------------------------------------
        # Validate latents
        # ------------------------------------------------------------
        self._check_latent(
            "latents_video",
            latents_video,
        )

        # Video-only ablation: contact/force are not modelled at all. They are
        # accepted in the signature (and ignored) so the pipeline can call this
        # with the same keyword arguments as the 3-modality model.

        # ------------------------------------------------------------
        # Validate timestep inputs
        # ------------------------------------------------------------
        if timestep_video is None:
            raise ValueError(
                "ACWMWanModelMoTForcing requires timestep_video."
            )



        # ------------------------------------------------------------
        # Validate action context
        # ------------------------------------------------------------
        if context is None:
            raise ValueError(
                "ACWMWanModelMoTForcing requires action context, got None."
            )

        if context.ndim != 3:
            raise ValueError(
                "ACWM action context must be "
                f"(B, T_action, dim), got {context.shape}."
            )

        if context.shape[-1] != self.dim:
            raise ValueError(
                f"ACWM action context must be projected to dim={self.dim}, "
                f"got last dim={context.shape[-1]}."
            )

        device = latents_video.device
        dtype = latents_video.dtype

        context = context.to(
            device=device,
            dtype=dtype,
        )

        # ------------------------------------------------------------
        # 1. Per-modality timestep embeddings
        #
        # IMPORTANT:
        #
        # We share self.time_embedding and self.time_projection weights.
        # Only the timestep VALUE differs between modalities.
        # ------------------------------------------------------------
        t_video, t_mod_video = self._build_time_condition(
            timestep_video,
            device=device,
            dtype=dtype,
        )

        t_mod_contact = None
        t_mod_force = None

        # ------------------------------------------------------------
        # 2. Patchify + concatenate modalities
        # ------------------------------------------------------------
        x, grid_size, token_counts = self.patchify_video_only(
            latents_video,
        )

        # ------------------------------------------------------------
        # 3. RoPE for concatenated tokens
        # ------------------------------------------------------------
        freqs = self.build_acwm_freqs(
            grid_size,
            device=x.device,
            num_modalities=1,
        )

        if freqs.shape[0] != x.shape[1]:
            raise ValueError(
                "RoPE/token length mismatch: "
                f"freqs={freqs.shape[0]}, "
                f"tokens={x.shape[1]}"
            )

        # ------------------------------------------------------------
        # 4. MoT blocks
        # ------------------------------------------------------------
        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x,
                    context,
                    t_mod_video,
                    t_mod_contact,
                    t_mod_force,
                    freqs,
                    token_counts,
                    grid_size,
                )

            else:
                x = block(
                    x,
                    context,
                    t_mod_video,
                    t_mod_contact,
                    t_mod_force,
                    freqs,
                    token_counts,
                    grid_size,
                )

        # ------------------------------------------------------------
        # 5. Final head with per-modality timestep conditioning
        #
        # self.head weights remain SHARED.
        #
        # We split the sequence and invoke the same head three times using
        # each modality's own time embedding.
        #
        # This is preferable to:
        #
        #     self.head(x, t_video)
        #
        # because that would incorrectly tell contact/force that they have
        # the video's timestep.
        # ------------------------------------------------------------
        x = self.head(
            x,
            t_video,
        )

        # ------------------------------------------------------------
        # 6. Split + unpatchify
        # ------------------------------------------------------------
        pred = {
            "video": self.unpatchify(x, grid_size),
        }

        return pred