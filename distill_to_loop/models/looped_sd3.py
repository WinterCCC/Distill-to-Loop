"""The looped, parameter-shared MMDiT student distilled from ``SD3Transformer2DModel``.

Structure (``K`` = the number of loop rounds, ``s_k`` the depth coordinate of round ``k``)::

    h, c = pos_embed(x_t), context_embedder(prompt_embeds)
    temb = time_text_embed(t, pooled)
    for block in pre:  (c, h) = block(h, c, temb)                          # teacher blocks 0 .. n_pre - 1
    for k in 1..K:
        (c', h') = loop blocks on (h, c) with temb + e_s(s_k)               # n_loop shared blocks
        h = h + alpha_k (h' - h);  c = c + alpha_k (c' - c)                 # alpha_k = s_k - s_{k-1}
        u_k = unpatchify(proj_out(norm_out(post(h, c, temb), temb)))       # readout; the main path is untouched
    return [u_k]

Only ``n_pre + n_loop + n_post`` physical :class:`JointTransformerBlock` instances exist, independent of ``K``: the shared blocks are re-entered
once per round and read out through the post blocks without disturbing the streams that continue into the next round.

The student options are fixed: a continuous depth embedding ``e_s`` (:class:`DepthEmbedding`, last layer zero at initialization) on the shared
blocks, the depth-step update ``alpha_k = s_k - s_{k-1}`` on both streams, a per-stream QK-norm on the shared and post blocks and soft XSA on the
shared blocks (:mod:`distill_to_loop.models.attention`). At inference the depth grid is uniform, ``s_k = k / K``, so every round moves the streams
by ``1 / K`` of its update; ``K`` is the mutable attribute :attr:`LoopedSD3Transformer.loop_rounds`.

``LoopedSD3Transformer`` subclasses ``SD3Transformer2DModel`` for its ``ConfigMixin`` / ``ModelMixin`` plumbing (a self-describing config,
``state_dict`` and ``from_config``), without its ``__init__``, which would build ``num_layers`` full blocks.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel, apply_lora_scale

from distill_to_loop.models.attention import STUDENT_ONLY_ATTN_KEYS, install_joint_attention_options

DEPTH_EMBEDDER_PREFIX = "depth_embedder."  # state_dict key prefix of the depth embedding e_s


# --------------------------------------------------------------------------- #
# depth embedding
# --------------------------------------------------------------------------- #
class DepthEmbedding(nn.Module):
    """``e_s(s) = Linear(SiLU(Linear(sinusoid(s))))`` of the depth coordinate ``s`` in ``(0, 1]``, last layer zero-initialized, so the embedding
    is the zero vector at initialization. The sinusoid is diffusers' ``Timesteps`` (``flip_sin_to_cos=True``, ``downscale_freq_shift=0``)."""

    def __init__(self, embedding_dim: int, num_freqs: int = 256):
        super().__init__()
        self.time_proj = Timesteps(num_channels=num_freqs, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.embedder = TimestepEmbedding(in_channels=num_freqs, time_embed_dim=embedding_dim)
        nn.init.zeros_(self.embedder.linear_2.weight)
        nn.init.zeros_(self.embedder.linear_2.bias)

    def forward(self, depth: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        proj = self.time_proj(depth)  # (batch, num_freqs), fp32
        return self.embedder(proj.to(dtype=dtype))


# --------------------------------------------------------------------------- #
# depth steps
# --------------------------------------------------------------------------- #
def step_alphas(depth_s: torch.Tensor) -> torch.Tensor:
    """``[B, K]`` ``alpha_k = s_k - s_{k-1}`` (``s_0 = 0``) of the depth coordinates ``depth_s`` (``[B, K]``), in their dtype."""
    return torch.diff(depth_s, dim=1, prepend=depth_s.new_zeros(depth_s.shape[0], 1))


def loop_step_weights(depth_s: torch.Tensor | None, n_rounds: int) -> list[float | torch.Tensor]:
    """The step weights ``alpha_k`` of the ``n_rounds`` rounds of one forward, round 1 first: the number ``1 / n_rounds`` for every round on the
    uniform grid (``depth_s=None``), else one ``[B, 1, 1]`` tensor per round (:func:`step_alphas`)."""
    if depth_s is None:
        return [1 / n_rounds] * n_rounds
    alpha = step_alphas(depth_s)
    return [alpha[:, k].view(-1, 1, 1) for k in range(n_rounds)]


def scale_round_update(h_prev: torch.Tensor, h_round: torch.Tensor, weight: float | torch.Tensor) -> torch.Tensor:
    """``h_prev + weight * (h_round - h_prev)``, a weight tensor cast to the dtype of ``h_round`` first."""
    if torch.is_tensor(weight):
        weight = weight.to(dtype=h_round.dtype)
    return h_prev + weight * (h_round - h_prev)


# --------------------------------------------------------------------------- #
# teacher -> student weight transfer
# --------------------------------------------------------------------------- #
def stride_loop_sources(num_layers: int, n_pre: int, n_loop: int, n_post: int) -> list[int]:
    """The teacher layer each shared block is initialized from: shared block ``j`` copies layer ``n_pre + j * (M / n_loop)``, ``M = num_layers -
    n_pre - n_post`` the teacher's middle segment (needs ``M % n_loop == 0``). The pre blocks copy the first ``n_pre`` layers and the post blocks
    the last ``n_post`` layers."""
    if n_pre + n_post > num_layers:
        raise ValueError(f"n_pre + n_post ({n_pre + n_post}) exceeds teacher num_layers ({num_layers})")
    m = num_layers - n_pre - n_post
    if m % n_loop != 0:
        raise ValueError(f"the stride initialization requires M % n_loop == 0 (M={m}, n_loop={n_loop})")
    step = m // n_loop
    return [n_pre + j * step for j in range(n_loop)]


def _copy_module_(dst: nn.Module, src: nn.Module) -> None:
    """Copy ``src``'s parameters and buffers into ``dst`` in place. ``dst`` may additionally hold the attention option parameters
    (:data:`~distill_to_loop.models.attention.STUDENT_ONLY_ATTN_KEYS`) that a teacher block lacks; those keep their initialization. Any other
    missing or unexpected key raises."""
    with torch.no_grad():
        result = dst.load_state_dict(src.state_dict(), strict=False)
    stray_missing = [k for k in result.missing_keys if k not in STUDENT_ONLY_ATTN_KEYS]
    if stray_missing or result.unexpected_keys:
        raise RuntimeError(
            f"state_dict mismatch copying {type(src).__name__} into {type(dst).__name__}: missing={stray_missing}, unexpected={list(result.unexpected_keys)}"
        )


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class LoopedSD3Transformer(SD3Transformer2DModel):
    """A depth-``n_pre + n_loop * K + n_post`` MMDiT with ``n_loop`` shared blocks (module docstring).

    Construct directly (random initialization, e.g. before loading exported weights) or with :meth:`from_teacher`. ``forward`` follows
    ``SD3Transformer2DModel.forward``'s calling convention, so the student can replace the transformer of a ``StableDiffusion3Pipeline``; it
    runs :attr:`loop_rounds` rounds (default ``n_rounds``, the trained count) and returns the last readout. Training calls
    :meth:`forward_rounds`, which returns every round's readout.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        qk_norm: str | None = None,
        n_pre: int = 2,
        n_loop: int = 5,
        n_post: int = 2,
        n_rounds: int = 4,
        depth_embed_freqs: int = 256,
    ):
        if n_pre < 0 or n_loop < 1 or n_post < 1:
            raise ValueError(f"require n_pre >= 0, n_loop >= 1, n_post >= 1; got {n_pre=}, {n_loop=}, {n_post=}")
        if n_rounds < 1:
            raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")

        # Skip SD3Transformer2DModel.__init__ (it would build `num_layers` full blocks); run the ModelMixin / ConfigMixin / nn.Module chain.
        super(SD3Transformer2DModel, self).__init__()

        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )
        self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)

        def _block(context_pre_only: bool, with_qk_norm: bool = False, xsa_soft: bool = False) -> JointTransformerBlock:
            block = JointTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                context_pre_only=context_pre_only,
                qk_norm=qk_norm,
            )
            install_joint_attention_options(block.attn, qk_norm=with_qk_norm, xsa_soft=xsa_soft)
            return block

        # The pre blocks run diffusers' attention; the shared blocks get QK-norm and soft XSA; the post blocks get QK-norm.
        self.pre = nn.ModuleList([_block(False) for _ in range(n_pre)])
        self.loop = nn.ModuleList([_block(False, with_qk_norm=True, xsa_soft=True) for _ in range(n_loop)])
        # Only the last post block is ever the final block of a forward pass, so only it is `context_pre_only`.
        self.post = nn.ModuleList([_block(i == n_post - 1, with_qk_norm=True) for i in range(n_post)])

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        # e_s(s_k), zero at initialization, read by the shared blocks only. Registered last.
        self.depth_embedder = DepthEmbedding(self.inner_dim, depth_embed_freqs)

        # Per-block activation checkpointing in training forwards (`_run_block`), off by default. Turn it on with `enable_gradient_checkpointing()`
        # (ModelMixin), which also sets `_gradient_checkpointing_func`; setting this attribute alone would leave that function unset.
        self.gradient_checkpointing = False
        # The number of rounds `forward` runs (K); mutable.
        self.loop_rounds = n_rounds
        # Round-level activation checkpointing of the shared blocks in training forwards; mutable, off by default.
        self.loop_grad_checkpoint = False

    # ----------------------------------------------------------------- #
    # construction from a pretrained teacher
    # ----------------------------------------------------------------- #
    @classmethod
    def from_teacher(
        cls, teacher: SD3Transformer2DModel, n_pre: int = 2, n_loop: int = 5, n_post: int = 2, n_rounds: int = 4,
    ) -> "LoopedSD3Transformer":
        """Build a student and initialize its weights from ``teacher``: the embeddings, ``norm_out`` and ``proj_out`` are copied, the pre and
        post blocks copy the teacher's first and last layers, and the shared blocks copy every ``M / n_loop``-th middle layer
        (:func:`stride_loop_sources`). The parameters a teacher lacks keep their initialization: the depth embedder (zero last layer), the
        QK-norm gains (ones, until the trainer calibrates them) and ``xsa_beta`` (zeros). ``teacher`` is not modified and not retained."""
        cfg = teacher.config
        if cfg.dual_attention_layers:
            raise NotImplementedError(
                "LoopedSD3Transformer does not support teachers with dual-attention blocks "
                f"(dual_attention_layers={tuple(cfg.dual_attention_layers)})"
            )
        num_layers = cfg.num_layers
        if n_pre + n_post > num_layers:
            raise ValueError(f"n_pre + n_post ({n_pre + n_post}) exceeds teacher num_layers ({num_layers})")

        student = cls(
            sample_size=cfg.sample_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            attention_head_dim=cfg.attention_head_dim,
            num_attention_heads=cfg.num_attention_heads,
            joint_attention_dim=cfg.joint_attention_dim,
            caption_projection_dim=cfg.caption_projection_dim,
            pooled_projection_dim=cfg.pooled_projection_dim,
            out_channels=cfg.out_channels,
            pos_embed_max_size=cfg.pos_embed_max_size,
            qk_norm=cfg.qk_norm,
            n_pre=n_pre,
            n_loop=n_loop,
            n_post=n_post,
            n_rounds=n_rounds,
        )

        _copy_module_(student.pos_embed, teacher.pos_embed)
        _copy_module_(student.time_text_embed, teacher.time_text_embed)
        _copy_module_(student.context_embedder, teacher.context_embedder)
        _copy_module_(student.norm_out, teacher.norm_out)
        _copy_module_(student.proj_out, teacher.proj_out)

        for j, block in enumerate(student.pre):
            _copy_module_(block, teacher.transformer_blocks[j])
        for j, block in enumerate(student.post):
            _copy_module_(block, teacher.transformer_blocks[num_layers - n_post + j])
        sources = stride_loop_sources(num_layers, n_pre, student.config.n_loop, n_post)
        for j, block in enumerate(student.loop):
            _copy_module_(block, teacher.transformer_blocks[sources[j]])
        return student

    # ----------------------------------------------------------------- #
    # forward
    # ----------------------------------------------------------------- #
    def _run_block(
        self,
        block: JointTransformerBlock,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb: torch.Tensor,
        joint_attention_kwargs: dict[str, Any] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            return self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, temb, joint_attention_kwargs
            )
        return block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    def _unpatchify(self, hidden_states: torch.Tensor, height: int, width: int) -> torch.Tensor:
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size
        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

    def _readout(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb: torch.Tensor,
        height: int,
        width: int,
        joint_attention_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        """Run ``post`` + ``norm_out`` + ``proj_out`` on a copy of the main path (the caller's tensors continue into the next round)."""
        h, c = hidden_states, encoder_hidden_states
        for block in self.post:
            c, h = self._run_block(block, h, c, temb, joint_attention_kwargs)
        h = self.norm_out(h, temb)
        h = self.proj_out(h)
        return self._unpatchify(h, height, width)

    def _round_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        depth_s: torch.Tensor,
        round_index: int,
        joint_attention_kwargs: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The shared block stack of round ``round_index`` (0-based) on both streams, conditioned on ``temb + e_s(s_k)``; returns
        ``(encoder_hidden_states, hidden_states)``. A function of its arguments and the module's parameters only, so :meth:`forward_rounds` may
        run it under activation checkpointing and the recomputation repeats exactly the operations of the forward."""
        temb_i = temb + self.depth_embedder(depth_s[:, round_index], dtype=temb.dtype)
        for block in self.loop:
            encoder_hidden_states, hidden_states = self._run_block(
                block, hidden_states, encoder_hidden_states, temb_i, joint_attention_kwargs
            )
        return encoder_hidden_states, hidden_states

    @apply_lora_scale("joint_attention_kwargs")
    def forward_rounds(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        n_rounds: int | None = None,
        readout: Literal["all", "last"] = "all",
        joint_attention_kwargs: dict[str, Any] | None = None,
        depth_s: torch.Tensor | None = None,
        include_prelude: bool = False,
        step_scale: bool = True,
    ) -> list[torch.Tensor]:
        """Run ``n_rounds`` rounds (default :attr:`loop_rounds`) and return the readouts: one per round with ``readout="all"``, the last one with
        ``"last"``.

        ``depth_s`` (``[B, n_rounds]`` float) is every sample's depth coordinate per round; ``None`` is the inference grid ``s_k = k / n_rounds``.
        ``include_prelude=True`` puts ``u_0`` first in the returned list: the pre blocks' output read out with no shared block run. Each round
        leaves ``h + alpha_k (R(h) - h)`` on the image stream and ``c + alpha_k (R_c(c) - c)`` on the text stream (``R`` / ``R_c`` the round's
        shared blocks), with ``alpha_k`` from ``depth_s`` per sample or ``1 / n_rounds`` on the grid; the readout and the next round read the
        updated streams. ``step_scale=False`` runs every round unscaled (``alpha_k = 1``): the forward the QK-norm gains are calibrated on.
        With :attr:`loop_grad_checkpoint` and gradient enabled, every round's shared block stack runs under non-reentrant activation
        checkpointing; the step update and the readouts stay outside the recomputed region.
        """
        if readout not in ("all", "last"):
            raise ValueError(f"readout must be 'all' or 'last', got {readout!r}")
        n_rounds = self.loop_rounds if n_rounds is None else n_rounds
        if n_rounds < 1:
            raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
        if depth_s is not None and tuple(depth_s.shape) != (hidden_states.shape[0], n_rounds):
            raise ValueError(f"depth_s must have shape (batch, n_rounds) = {(hidden_states.shape[0], n_rounds)}, got {tuple(depth_s.shape)}")
        # The step weights are known numbers on the grid (depth_s None) and per-sample tensors for given coordinates.
        step_weights = loop_step_weights(depth_s, n_rounds) if step_scale else None
        if depth_s is None:  # the grid the depth embedder reads
            depth_s = torch.arange(1, n_rounds + 1, device=hidden_states.device, dtype=torch.float32).div_(n_rounds).expand(hidden_states.shape[0], n_rounds)

        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states)
        hidden_states = hidden_states.contiguous()
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for block in self.pre:
            encoder_hidden_states, hidden_states = self._run_block(
                block, hidden_states, encoder_hidden_states, temb, joint_attention_kwargs
            )

        outputs: list[torch.Tensor] = []
        if include_prelude:  # u_0: the pre blocks' output straight through the readout, no shared block
            outputs.append(self._readout(hidden_states, encoder_hidden_states, temb, height, width, joint_attention_kwargs))
        for i in range(n_rounds):
            h_prev, c_prev = hidden_states, encoder_hidden_states  # the streams this round starts from
            if self.loop_grad_checkpoint and torch.is_grad_enabled():  # recompute the round's block activations in the backward pass
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    self._round_blocks, hidden_states, encoder_hidden_states, temb, depth_s, i, joint_attention_kwargs, use_reentrant=False,
                )
            else:
                encoder_hidden_states, hidden_states = self._round_blocks(
                    hidden_states, encoder_hidden_states, temb, depth_s, i, joint_attention_kwargs
                )
            if step_weights is not None:  # the same alpha_k on both streams, before the readout and the next round
                hidden_states = scale_round_update(h_prev, hidden_states, step_weights[i])
                encoder_hidden_states = scale_round_update(c_prev, encoder_hidden_states, step_weights[i])
            if readout == "all" or i == n_rounds - 1:
                outputs.append(
                    self._readout(hidden_states, encoder_hidden_states, temb, height, width, joint_attention_kwargs)
                )
        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: list | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        skip_layers: list[int] | None = None,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """``SD3Transformer2DModel.forward``-compatible entry point for sampling pipelines: :attr:`loop_rounds` rounds on the uniform depth grid,
        the last readout only. ``skip_layers`` and ``block_controlnet_hidden_states`` are accepted for signature compatibility only and raise when
        given."""
        if skip_layers is not None:
            raise NotImplementedError("LoopedSD3Transformer does not support skip_layers (skip-layer guidance)")
        if block_controlnet_hidden_states is not None:
            raise NotImplementedError("LoopedSD3Transformer does not support block_controlnet_hidden_states (ControlNet)")
        outputs = self.forward_rounds(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            n_rounds=self.loop_rounds,
            readout="last",
            joint_attention_kwargs=joint_attention_kwargs,
        )
        output = outputs[0]
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
