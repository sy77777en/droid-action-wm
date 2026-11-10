# diffsynth/models/wan_video_dit_acwm_mot.py
"""
ACWM Mixture-of-Transformers (MoT) DiT.

Granularity: per-modality q/k/v/o projections AND per-modality FFN, but the
self-attention *computation* is full-sequence (video/contact/force tokens all
attend to each other after their per-modality projections are concatenated).

  self-attn:
      x_v -> q_v,k_v,v_v ; x_c -> q_c,k_c,v_c ; x_f -> q_f,k_f,v_f
      Q = cat(q_v,q_c,q_f) ; K = cat(...) ; V = cat(...)     # full sequence
      RoPE applied per-modality-segment (each segment uses its own freqs slice,
      which are identical copies -> same as building 3x-repeated freqs)
      attn(Q,K,V)  # cross-modal, full sequence
      split back -> o_v / o_c / o_f
  cross-attn:
      each modality has its own q/k/v/o attending the SHARED action context
  ffn:
      each modality has its own FFN
  modulation:
      each modality has its own learnable modulation base (t_mod is shared,
      since timestep is global)

Everything else (patchify / split / unpatchify / RoPE construction / loss /
pipeline) is reused from ACWMWanModel.

Init: all three per-modality copies are deep-copied from the original shared
DiTBlock submodules, so at step 0 the MoT model is functionally identical to
the shared-backbone ACWMWanModel (fair ablation start point).
"""

import copy
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn

# Action cross-attention mode. Default (unset) is framewise, which is what every
# trained checkpoint so far uses -- do not change the default.
#
# Why this exists: framewise action cross-attention arrived in the same commit as
# the MoT block and has no counterpart in the pre-MoT base model
# (wan_video_dit_acwm.py), which feeds the action in as `context` to the stock
# DiTBlock and therefore does global cross-attention. Comparing MoT against that
# base would confound "MoT" with "framewise conditioning". Setting
# ACWM_MOT_GLOBAL_CROSS_ATTN=1 puts MoT on the base's global path so the
# comparison isolates MoT.
#
# Read once at import, not per forward: flipping it mid-process would silently
# produce a model whose blocks disagree with its checkpoint.
USE_GLOBAL_CROSS_ATTN = os.environ.get("ACWM_MOT_GLOBAL_CROSS_ATTN", "0") == "1"
if USE_GLOBAL_CROSS_ATTN:
    print("[ACWM-MoT] ACWM_MOT_GLOBAL_CROSS_ATTN=1 -> global action cross-attention "
          "(ablation mode; checkpoints are NOT interchangeable with framewise ones)",
          flush=True)

from .wan_video_dit import (
    sinusoidal_embedding_1d,
    modulate,
    rope_apply,
    flash_attention,
    DiTBlock,
)
from .wan_video_dit_acwm import ACWMWanModel
from ..core.gradient import gradient_checkpoint_forward


MODALITIES = ("video", "contact", "force")


def _split3(x, token_counts):
    nv = token_counts["video"]
    nc = token_counts["contact"]
    nf = token_counts["force"]
    return x[:, :nv], x[:, nv:nv + nc], x[:, nv + nc:nv + nc + nf]


def _split_freqs3(freqs, token_counts):
    # freqs is (Nv+Nc+Nf, 1, rope_dim); split the same way as tokens
    nv = token_counts["video"]
    nc = token_counts["contact"]
    nf = token_counts["force"]
    return freqs[:nv], freqs[nv:nv + nc], freqs[nv + nc:nv + nc + nf]


class MoTDiTBlock(nn.Module):
    """MoT block: per-modality self-attn q/k/v/o, cross-attn q/k/v/o, ffn, and
    modulation. Self-attention is computed over the full concatenated sequence.

    Built by deep-copying the submodules of an existing shared `DiTBlock`, so
    all three modality branches start identical to the pretrained block."""

    def __init__(self, base_block: DiTBlock):
        super().__init__()
        self.dim = base_block.dim
        self.num_heads = base_block.num_heads
        self.ffn_dim = base_block.ffn_dim

        # shared norms are parameter-free (elementwise_affine=False for norm1/2;
        # norm3 has affine but we keep ONE shared copy to limit param blow-up —
        # can be made per-modality later if needed). We deep-copy per modality
        # for the parameterized parts: self_attn, cross_attn, ffn, modulation.
        for m in MODALITIES:
            setattr(self, f"self_attn_{m}", copy.deepcopy(base_block.self_attn))
            setattr(self, f"ffn_{m}", copy.deepcopy(base_block.ffn))
            # per-modality learnable modulation base (1,6,dim)
            setattr(self, f"modulation_{m}",
                    nn.Parameter(base_block.modulation.detach().clone()))

        # Cross-attention: q/o are per-modality (v/c/f mixture), but k/v come
        # from the SHARED action embedding — action is NOT a mixture modality,
        # so it is projected to k/v ONCE and all three modalities attend it.
        # We keep one shared cross_attn for its k/v/norm_k/norm_q, and add
        # per-modality q/o projections.
        self.cross_attn = copy.deepcopy(base_block.cross_attn)   # provides k,v,norm_k
        for m in MODALITIES:
            setattr(self, f"cross_q_{m}", copy.deepcopy(base_block.cross_attn.q))
            setattr(self, f"cross_o_{m}", copy.deepcopy(base_block.cross_attn.o))
            setattr(self, f"cross_norm_q_{m}", copy.deepcopy(base_block.cross_attn.norm_q))

        # norms: norm1/norm2 are affine-free -> shareable & cheap; keep shared.
        # norm3 (pre cross-attn) has affine; keep shared copy.
        self.norm1 = copy.deepcopy(base_block.norm1)
        self.norm2 = copy.deepcopy(base_block.norm2)
        self.norm3 = copy.deepcopy(base_block.norm3)
        self.gate = copy.deepcopy(base_block.gate)

    # ---- self-attn helpers: reuse the q/k/v/o/norm_q/norm_k of each branch ----
    def _sa_project(self, sa, x, freqs):
        # returns q,k,v with RoPE applied to q,k (v raw), matching SelfAttention
        q = sa.norm_q(sa.q(x))
        k = sa.norm_k(sa.k(x))
        v = sa.v(x)
        q = rope_apply(q, freqs, sa.num_heads)
        k = rope_apply(k, freqs, sa.num_heads)
        return q, k, v

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
                Number of raw action tokens assigned to one latent frame.
    
        Returns:
            (B, T_action // actions_per_latent, actions_per_latent, D)
        """
        if context.ndim != 3:
            raise ValueError(
                f"Expected context shape (B, T_action, D), got {context.shape}."
            )

        B, seq_action, D = context.shape

        if seq_action % actions_per_latent != 0:
            raise ValueError(
                f"Action sequence length {seq_action} is not divisible by "
                f"actions_per_latent={actions_per_latent}."
            )
    
        num_action_latents = seq_action // actions_per_latent
    
        return context.reshape(
            B,
            num_action_latents,
            actions_per_latent,
            D,
        )

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
        corresponding action groups. Earlier observation/history frames receive
        zero action cross-attention output.
        """
        f, h, w = grid_size
        B, N, D = x.shape
        spatial_tokens = h * w
    
        if N != f * spatial_tokens:
            raise ValueError(
                f"{modality}: N={N}, expected {f * spatial_tokens}, "
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

        #  if not hasattr(self, "_framewise_verified"):
        import sys
        # print(
            # "[Framewise verified]",
            # f"grid_size={(f, h, w)}",
            # f"x={tuple(x.shape)}",
            # f"context={tuple(action_context.shape)}",
            # f"action_groups={tuple(action_groups.shape)}",
            # f"prefix_frames={prefix_frames}",
            # f"modality={modality}",
            # file=sys.stderr,
            # flush=True,
        # )
           #  self._framewise_verified = True
    
        # (B, f*h*w, D) -> (B, f, h*w, D)
        x_framewise = x.reshape(
            B,
            f,
            spatial_tokens,
            D,
        )
    
        # Only the final num_action_latents frames receive action conditioning.
        x_action = x_framewise[:, prefix_frames:]
    
        q_proj = getattr(self, f"cross_q_{modality}")
        q_norm = getattr(self, f"cross_norm_q_{modality}")
        out_proj = getattr(self, f"cross_o_{modality}")
    
        q = q_norm(
            q_proj(
                self.norm3(x_action)
            )
        )
    
        ca = self.cross_attn
        k = ca.norm_k(ca.k(action_groups))
        v = ca.v(action_groups)
    
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
    
        action_out = ca.attn(q, k, v)
        action_out = out_proj(action_out)
    
        action_out = action_out.reshape(
            B,
            num_action_latents,
            spatial_tokens,
            D,
        )
    
        # Observation/history frames receive no future-action cross-attention.
        prefix_out = torch.zeros(
            B,
            prefix_frames,
            spatial_tokens,
            D,
            dtype=action_out.dtype,
            device=action_out.device,
        )
    
        out = torch.cat(
            [prefix_out, action_out],
            dim=1,
        )
    
        return out.reshape(B, N, D)
    

    def forward(self, x, context, t_mod, freqs, token_counts, grid_size):
        # ---- modulation (per modality, t_mod shared) ----
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        x_v, x_c, x_f = _split3(x, token_counts)
        fq_v, fq_c, fq_f = _split_freqs3(freqs, token_counts)

        branches = {
            "video": (x_v, fq_v),
            "contact": (x_c, fq_c),
            "force": (x_f, fq_f),
        }

        mod = {}
        for m in MODALITIES:
            base = getattr(self, f"modulation_{m}").to(dtype=t_mod.dtype, device=t_mod.device)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                (base + t_mod).chunk(6, dim=chunk_dim)
            if has_seq:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                    shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
                )
            mod[m] = (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)

        # ---- self-attention: per-modality qkv -> full-seq attention -> per-modality o ----
        q_list, k_list, v_list = [], [], []
        for m in MODALITIES:
            xm, fqm = branches[m]
            shift_msa, scale_msa = mod[m][0], mod[m][1]
            xin = modulate(self.norm1(xm), shift_msa, scale_msa)
            sa = getattr(self, f"self_attn_{m}")
            q, k, v = self._sa_project(sa, xin, fqm)
            q_list.append(q); k_list.append(k); v_list.append(v)

        Q = torch.cat(q_list, dim=1)
        K = torch.cat(k_list, dim=1)
        V = torch.cat(v_list, dim=1)

        # full-sequence attention (cross-modal). num_heads identical across modalities.
        attn_out = flash_attention(Q, K, V, num_heads=self.num_heads)

        a_v, a_c, a_f = _split3(attn_out, token_counts)
        attn_by_m = {"video": a_v, "contact": a_c, "force": a_f}

        new_x = {}
        for m in MODALITIES:
            xm, _ = branches[m]
            sa = getattr(self, f"self_attn_{m}")
            gate_msa = mod[m][2]
            o = sa.o(attn_by_m[m])
            new_x[m] = self.gate(xm, gate_msa, o)

        if USE_GLOBAL_CROSS_ATTN:
            # ---- global cross-attention: per-modality q/o, SHARED k/v from action ----
            # action context is projected to k/v ONCE (shared). Each modality has
            # its own q and o. Queries are concatenated to full sequence, attend the
            # shared action k/v, then split back to per-modality o.
            #
            # Every latent frame sees every action. This is what the pre-MoT base
            # model (ACWMWanModel + stock DiTBlock) does, so selecting it here makes
            # the MoT-vs-base comparison differ only in MoT. See the module header.
            ca = self.cross_attn
            k = ca.norm_k(ca.k(context))
            v = ca.v(context)

            q_list = []
            for m in MODALITIES:
                xm = new_x[m]
                nq = getattr(self, f"cross_norm_q_{m}")
                wq = getattr(self, f"cross_q_{m}")
                q_list.append(nq(wq(self.norm3(xm))))
            Qc = torch.cat(q_list, dim=1)

            ca_out = ca.attn(Qc, k, v)              # AttentionModule(num_heads)
            c_v, c_c, c_f = _split3(ca_out, token_counts)
            ca_by_m = {"video": c_v, "contact": c_c, "force": c_f}

            for m in MODALITIES:
                wo = getattr(self, f"cross_o_{m}")
                new_x[m] = new_x[m] + wo(ca_by_m[m])
        else:
            # ---- framewise cross-attention: per-modality q/o, SHARED k/v from action ----
            # Each future latent frame sees only its corresponding four actions:
            #
            # latent 0: no action
            # latent 1: actions 0-3
            # latent 2: actions 4-7
            # latent 3: actions 8-11
            # latent 4: actions 12-15
            for m in MODALITIES:
                framewise_out = self._framewise_action_cross_attention(
                    x=new_x[m],
                    action_context=context,
                    grid_size=grid_size,
                    modality=m,
                )

                new_x[m] = new_x[m] + framewise_out

        # ---- FFN (per-modality) ----
        for m in MODALITIES:
            xm = new_x[m]
            shift_mlp, scale_mlp, gate_mlp = mod[m][3], mod[m][4], mod[m][5]
            ffn = getattr(self, f"ffn_{m}")
            xin = modulate(self.norm2(xm), shift_mlp, scale_mlp)
            xm = self.gate(xm, gate_mlp, ffn(xin))
            new_x[m] = xm

        # ---- concat back in canonical order ----
        return torch.cat([new_x["video"], new_x["contact"], new_x["force"]], dim=1)


class ACWMWanModelMoT(ACWMWanModel):
    """ACWM DiT with MoT blocks. Reuses ACWMWanModel's patchify/split/RoPE and
    only replaces the shared DiTBlocks with per-modality MoTDiTBlocks."""

    _repeated_blocks = ["MoTDiTBlock"]

    def __init__(self, *args, source_blocks=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Build MoT blocks by deep-copying from `source_blocks` if provided
        # (these should be the pretrained-weight-loaded shared DiTBlocks of an
        # already-converted ACWMWanModel), otherwise from this model's own
        # freshly-initialized shared blocks (used only for standalone tests).
        src = source_blocks if source_blocks is not None else self.blocks
        if len(src) != len(self.blocks):
            raise ValueError(
                f"source_blocks length {len(src)} != num_layers {len(self.blocks)}.")
        mot_blocks = nn.ModuleList()
        for block in src:
            mot_blocks.append(MoTDiTBlock(block))
        self.blocks = mot_blocks

    def init_mot_from_shared(self):
        """No-op placeholder: MoT blocks are already initialized from the
        shared blocks at construction. Kept for API symmetry with
        init_physical_patch_embeddings_from_video()."""
        pass

    # ------------------------------------------------------------------ #
    # forward: same as ACWMWanModel but blocks take token_counts
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
        self._check_latent("latents_video", latents_video)
        self._check_latent("latents_contact", latents_contact)
        self._check_latent("latents_force", latents_force)
        self._check_same_shape(latents_video, latents_contact, latents_force)

        if context is None:
            raise ValueError("ACWMWanModelMoT requires action context, got None.")
        if context.ndim != 3:
            raise ValueError(
                f"ACWM action context must be (B, T_action, dim), got {context.shape}.")
        if context.shape[-1] != self.dim:
            raise ValueError(
                f"ACWM action context must be projected to dim={self.dim}, "
                f"got last dim={context.shape[-1]}.")

        device = latents_video.device
        dtype = latents_video.dtype
        context = context.to(device=device, dtype=dtype)
        timestep = timestep.to(dtype=torch.float32)

        # 1. timestep embedding
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(device=device, dtype=dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # 2. patchify + concat
        x, grid_size, token_counts = self.patchify_acwm(
            latents_video, latents_contact, latents_force)

        # 3. RoPE for concatenated tokens
        freqs = self.build_acwm_freqs(grid_size, device=x.device, num_modalities=3)
        if freqs.shape[0] != x.shape[1]:
            raise ValueError(
                f"RoPE/token length mismatch: freqs={freqs.shape[0]}, tokens={x.shape[1]}")

        # 4. MoT blocks (need token_counts to split per modality)
        for block in self.blocks:
            if self.training:
                # x = gradient_checkpoint_forward(
                #     block,
                #     use_gradient_checkpointing,
                #     use_gradient_checkpointing_offload,
                #     x, context, t_mod, freqs, token_counts,
                # )
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs, token_counts,
                    grid_size,
                )
            else:
                x = block(x, context, t_mod, freqs, token_counts, grid_size)

        # 5. shared head + split/unpatchify
        x = self.head(x, t)
        pred = self.unpatchify_acwm(x, grid_size, token_counts)
        return pred
