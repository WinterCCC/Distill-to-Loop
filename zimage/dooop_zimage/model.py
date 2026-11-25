"""The looped, parameter-shared student of Z-Image's ``ZImageTransformer2DModel`` (text to image).

The teacher is a single-stream DiT: the image tokens go through ``noise_refiner`` (modulated by ``adaln_input = t_embedder(1000 t)``), the
caption tokens through ``cap_embedder`` and ``context_refiner`` (no modulation); both are joined as ``[image, caption]`` and run through
``n_layers`` = 30 main blocks (each with its own ``adaLN_modulation`` on ``adaln_input``) and ``FinalLayer`` (also on ``adaln_input``). The
student replaces the 30 main blocks by ``n_pre`` + ``n_loop`` x K + ``n_post`` physical blocks (``K`` = the number of loop rounds, ``s_k`` the
depth coordinate of round ``k``)::

    u = [noise_refiner(x_tokens), context_refiner(cap_tokens)];  a = t_embedder(1000 t)        # the teacher's front end, unchanged
    for block in pre:   u = block(u, a)                                                        # teacher layers 0 .. n_pre - 1
    for k in 1..K:
        u = u + alpha_k (R_k(u) - u)     R_k: the n_loop shared blocks, each reading a + e_s(s_k);  alpha_k = s_k - s_{k-1}
        y_k = FinalLayer(post(u), a)                                                           # readout; the main path continues from u

Only ``n_pre + n_loop + n_post`` physical ``ZImageTransformerBlock`` instances exist, independent of ``K``. The shared blocks are initialized
from every ``M / n_loop``-th teacher layer of the middle segment (``M = n_layers - n_pre - n_post``; for 3 + 6 + 3: layers 3, 7, 11, 15, 19,
23), the pre blocks from layers ``0 .. n_pre - 1`` and the post blocks from the last ``n_post`` layers.

The depth embedding ``e_s(s_k)`` (:class:`~dooop.models.looped_sd3.DepthEmbedding`, 256 wide, last layer zero at initialization) is
added to the ``adaln_input`` the shared blocks read and nowhere else. The teacher's q and k RMSNorm layers are kept; no other attention option
is added. At inference the depth grid is uniform, ``s_k = k / K``, so every round moves the tokens by ``1 / K`` of its update.

Time convention. ``ZImagePipeline`` hands the transformer ``(1000 - t) / 1000 = 1 - sigma`` (``t`` the scheduler timestep ``sigma * 1000``)
and negates its output: ``noise_pred = -model_out`` is the velocity ``noise - x0``. :meth:`LoopedZImageTransformer.forward` has the teacher's
signature, takes ``1 - sigma`` and returns the raw (un-negated) output, so ``ZImagePipeline(transformer=student)`` samples with it. The training
paths (:meth:`~LoopedZImageTransformer.forward_rounds`, :class:`ZImageTeacherAdapter`) take the scheduler timestep ``sigma * 1000``, convert it
with the pipeline's own arithmetic (:func:`model_time`) and return the velocity ``-raw`` as ``[B, C, H, W]``.

Captions. The teacher takes a list of ``[len_i, 2560]`` captions (a caption's length moves the RoPE positions of the image tokens, so a
zero-padded batch is not equivalent). The training interface carries one fixed-width text row per sample; the caption's validity mask (1 for a
real token) travels in the ``pooled_projections`` slot (``[B, L]``), which caption dropout, the conditional / unconditional concatenation and
the rollout's repeats already handle, and :func:`caption_rows` hands the model the valid rows only.

The forward repeats private methods of diffusers 0.40.0's ``ZImageTransformer2DModel`` (its basic text-to-image path), so constructing the
student under another diffusers version is refused (:data:`DIFFUSERS_VERSION_WRITTEN_AGAINST`).
"""

from __future__ import annotations

import inspect
from typing import Any, Literal

import diffusers
import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_z_image import ADALN_EMBED_DIM, ZImageTransformer2DModel, ZImageTransformerBlock

from dooop.models.looped_sd3 import DepthEmbedding, _copy_module_, loop_step_weights, scale_round_update, stride_loop_sources

DIFFUSERS_VERSION_WRITTEN_AGAINST = "0.40.0"  # the ZImageTransformer2DModel whose forward the student repeats
# The constructor arguments of ZImageTransformer2DModel (the teacher's config fields), read off its signature so a student records every one.
TEACHER_CONFIG_FIELDS: tuple[str, ...] = tuple(name for name in inspect.signature(ZImageTransformer2DModel.__init__).parameters if name != "self")
# What from_teacher copies besides the main blocks: the teacher's front end and head (modules) and its two pad tokens (bare parameters).
COPIED_MODULES = ("all_x_embedder", "all_final_layer", "noise_refiner", "context_refiner", "t_embedder", "cap_embedder")
COPIED_PARAMETERS = ("x_pad_token", "cap_pad_token")
PIPELINE_TIME_SCALE = 1000.0  # ZImagePipeline's (1000 - t) / 1000


def require_diffusers_version_written_against() -> None:
    """Raise ``RuntimeError`` unless the installed diffusers is :data:`DIFFUSERS_VERSION_WRITTEN_AGAINST`."""
    if diffusers.__version__ != DIFFUSERS_VERSION_WRITTEN_AGAINST:
        raise RuntimeError(
            f"LoopedZImageTransformer repeats the forward of diffusers {DIFFUSERS_VERSION_WRITTEN_AGAINST}'s ZImageTransformer2DModel, but diffusers "
            f"{diffusers.__version__} is installed; install diffusers=={DIFFUSERS_VERSION_WRITTEN_AGAINST} (requirements.txt)"
        )


def model_time(timestep: torch.Tensor) -> torch.Tensor:
    """The transformer's time input ``(1000 - t) / 1000 = 1 - sigma`` of a scheduler timestep ``t = sigma * 1000``: the arithmetic of
    ``ZImagePipeline.__call__``, op for op."""
    return (PIPELINE_TIME_SCALE - timestep) / PIPELINE_TIME_SCALE


def caption_rows(prompt_embeds: torch.Tensor, valid: torch.Tensor) -> list[torch.Tensor]:
    """Fixed-width text ``[B, L, D]`` as the teacher's list of ``[len_i, D]`` captions, ``valid`` ``[B, L]`` the 0/1 mask of the real tokens
    (any float dtype). ``ValueError`` for a row without a valid token."""
    keep = valid > 0.5
    rows = [embeds[mask] for embeds, mask in zip(prompt_embeds, keep)]
    if any(r.shape[0] == 0 for r in rows):
        raise ValueError("a caption has no valid token: the validity mask (the pooled_projections slot) is all zero")
    return rows


def image_list(hidden_states: torch.Tensor) -> list[torch.Tensor]:
    """``[B, C, H, W]`` latents as the teacher's list of ``[C, 1, H, W]`` (``ZImagePipeline``: ``latents.unsqueeze(2).unbind(0)``)."""
    return list(hidden_states.unsqueeze(2).unbind(0))


def velocity_batch(raw: list[torch.Tensor]) -> torch.Tensor:
    """The teacher's list of ``[C, 1, H, W]`` outputs as the velocity ``-raw`` ``[B, C, H, W]`` (``ZImagePipeline``: ``-model_out``)."""
    return -torch.stack(raw, dim=0).squeeze(2)


class LoopedZImageTransformer(ZImageTransformer2DModel):
    """A depth-``n_pre + n_loop * K + n_post`` Z-Image transformer with ``n_loop`` shared blocks (module docstring).

    Construct directly (random initialization, e.g. before loading exported weights) or with :meth:`from_teacher`. :meth:`forward_rounds`
    returns every round's velocity readout (training); :meth:`forward` has ``ZImageTransformer2DModel.forward``'s signature and returns the raw
    output of :attr:`loop_rounds` rounds on the grid ``s_k = k / K``. ``loop_rounds`` (default ``n_rounds``, the trained count) and
    ``loop_grad_checkpoint`` (round-level activation checkpointing of the shared blocks, default off) are mutable attributes; per-block
    checkpointing is diffusers' ``enable_gradient_checkpointing()``.
    """

    @register_to_config
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        siglip_feat_dim=None,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
        n_pre: int = 3,
        n_loop: int = 6,
        n_post: int = 3,
        n_rounds: int = 4,
        depth_embed_freqs: int = 256,
    ):
        require_diffusers_version_written_against()
        if n_pre < 0 or n_loop < 1 or n_post < 0 or n_rounds < 1:
            raise ValueError(f"require n_pre >= 0, n_loop >= 1, n_post >= 0, n_rounds >= 1; got {n_pre=}, {n_loop=}, {n_post=}, {n_rounds=}")
        if siglip_feat_dim is not None:
            raise ValueError(f"the student is the text-to-image model: siglip_feat_dim must be null, got {siglip_feat_dim!r}")
        # The teacher's own constructor (undecorated: the decorator would register n_layers = 0 over this class's config) builds the front end,
        # the head, the pad tokens and the RoPE embedder with no main block; the main blocks are then the student's pre / loop / post lists.
        ZImageTransformer2DModel.__init__.__wrapped__(
            self, all_patch_size=all_patch_size, all_f_patch_size=all_f_patch_size, in_channels=in_channels, dim=dim, n_layers=0,
            n_refiner_layers=n_refiner_layers, n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=norm_eps, qk_norm=qk_norm,
            cap_feat_dim=cap_feat_dim, siglip_feat_dim=None, rope_theta=rope_theta, t_scale=t_scale, axes_dims=axes_dims, axes_lens=axes_lens,
        )
        del self.layers

        def _block(layer_id: int) -> ZImageTransformerBlock:
            return ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)

        self.pre = nn.ModuleList([_block(i) for i in range(n_pre)])
        self.loop = nn.ModuleList([_block(n_pre + j) for j in range(n_loop)])
        self.post = nn.ModuleList([_block(n_pre + n_loop + i) for i in range(n_post)])
        # e_s(s_k), zero at initialization, read by the shared blocks only.
        self.depth_embedder = DepthEmbedding(min(dim, ADALN_EMBED_DIM), depth_embed_freqs)

        # enable_gradient_checkpointing() (ModelMixin) also sets _gradient_checkpointing_func; the attribute alone would leave it unset.
        self.gradient_checkpointing = False
        self.loop_rounds = n_rounds
        self.loop_grad_checkpoint = False

    # ----------------------------------------------------------------- #
    # construction from the teacher
    # ----------------------------------------------------------------- #
    @classmethod
    def from_teacher(
        cls, teacher: ZImageTransformer2DModel, n_pre: int = 3, n_loop: int = 6, n_post: int = 3, n_rounds: int = 4,
    ) -> "LoopedZImageTransformer":
        """A student with ``teacher``'s config, its weights copied from ``teacher`` (which is neither modified nor retained): the front end, head
        and pad tokens (:data:`COPIED_MODULES`, :data:`COPIED_PARAMETERS`), the pre blocks from layers ``0 .. n_pre - 1``, the post blocks from the
        last ``n_post`` layers and the shared blocks from the layers :func:`~dooop.models.looped_sd3.stride_loop_sources` names. The
        depth embedder keeps its zero-initialized last layer."""
        config = {name: teacher.config[name] for name in TEACHER_CONFIG_FIELDS}
        num_layers = int(config["n_layers"])
        loop_sources = stride_loop_sources(num_layers, n_pre, n_loop, n_post)
        student = cls(**config, n_pre=n_pre, n_loop=n_loop, n_post=n_post, n_rounds=n_rounds)
        for name in COPIED_MODULES:
            _copy_module_(getattr(student, name), getattr(teacher, name))
        with torch.no_grad():
            for name in COPIED_PARAMETERS:
                getattr(student, name).copy_(getattr(teacher, name))
        for blocks, layers in ((student.pre, range(n_pre)), (student.post, range(num_layers - n_post, num_layers))):
            for block, layer in zip(blocks, layers, strict=True):
                _copy_module_(block, teacher.layers[layer])
        for block, layer in zip(student.loop, loop_sources, strict=True):
            _copy_module_(block, teacher.layers[layer])
        return student

    # ----------------------------------------------------------------- #
    # forward pieces
    # ----------------------------------------------------------------- #
    def _call(self, layer: nn.Module, *args: Any) -> torch.Tensor:
        """``layer(*args)``, through diffusers' per-block checkpointing when it is on and gradients are tracked (as the teacher's forward)."""
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            return self._gradient_checkpointing_func(layer, *args)
        return layer(*args)

    def _front_end(self, x: list[torch.Tensor], t: torch.Tensor, cap_feats: list[torch.Tensor], patch_size: int, f_patch_size: int):
        """The teacher's forward up to the main blocks (its basic mode, line for line): ``(unified, unified_freqs, unified_mask, adaln_input,
        x_size)``."""
        device = x[0].device
        adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0])
        x, cap_feats, x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)
        x_seqlens = [len(xi) for xi in x]
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x, dim=0))
        x, x_freqs, x_mask, _, _ = self._prepare_sequence(list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask, self.x_pad_token, None, device)
        for layer in self.noise_refiner:
            x = self._call(layer, x, x_mask, x_freqs, adaln_input, None, None, None)
        cap_seqlens = [len(ci) for ci in cap_feats]
        cap_feats = self.cap_embedder(torch.cat(cap_feats, dim=0))
        cap_feats, cap_freqs, cap_mask, _, _ = self._prepare_sequence(
            list(cap_feats.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, self.cap_pad_token, None, device
        )
        for layer in self.context_refiner:
            cap_feats = self._call(layer, cap_feats, cap_mask, cap_freqs)
        unified, unified_freqs, unified_mask, _ = self._build_unified_sequence(
            x, x_freqs, x_seqlens, None, cap_feats, cap_freqs, cap_seqlens, None, None, None, None, None, False, device
        )
        return unified, unified_freqs, unified_mask, adaln_input, x_size

    def _round_blocks(self, unified: torch.Tensor, mask: torch.Tensor | None, freqs: torch.Tensor, adaln_input: torch.Tensor,
                      depth_s: torch.Tensor, round_index: int) -> torch.Tensor:
        """The shared block stack of round ``round_index`` (0-based) on ``adaln_input + e_s(s_k)``, formed here: a function of its arguments and
        the parameters only, so ``loop_grad_checkpoint``'s recomputation repeats exactly the forward's operations."""
        round_input = adaln_input + self.depth_embedder(depth_s[:, round_index], dtype=adaln_input.dtype)
        for block in self.loop:
            unified = self._call(block, unified, mask, freqs, round_input, None, None, None)
        return unified

    def _readout(self, unified: torch.Tensor, mask: torch.Tensor | None, freqs: torch.Tensor, adaln_input: torch.Tensor, x_size: list,
                 patch_size: int, f_patch_size: int) -> list[torch.Tensor]:
        """The post blocks, ``FinalLayer`` and the unpatchify of the teacher's forward on the main path's tokens, which the caller keeps: the raw
        output as the teacher's list of ``[C, 1, H, W]``. Every module here reads the plain ``adaln_input``."""
        for block in self.post:
            unified = self._call(block, unified, mask, freqs, adaln_input, None, None, None)
        out = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, c=adaln_input)
        return self.unpatchify(list(out.unbind(dim=0)), x_size, patch_size, f_patch_size, None)

    def _run_rounds(
        self, x: list[torch.Tensor], t: torch.Tensor, cap_feats: list[torch.Tensor], n_rounds: int | None, readout: str, depth_s: torch.Tensor | None,
        include_prelude: bool, patch_size: int = 2, f_patch_size: int = 1,
    ) -> list[list[torch.Tensor]]:
        """The raw readouts (the teacher's list of ``[C, 1, H, W]`` per sample) of the prelude (the pre blocks' output, with ``include_prelude``)
        and of every round (``readout="all"``) or the last (``"last"``). ``t`` is the transformer's time input ``1 - sigma``; ``depth_s``
        (``[B, n_rounds]``) the per-sample depth coordinates, ``None`` for the uniform grid ``s_k = k / n_rounds``."""
        assert patch_size in self.all_patch_size and f_patch_size in self.all_f_patch_size
        if readout not in ("all", "last"):
            raise ValueError(f"readout must be 'all' or 'last', got {readout!r}")
        n_rounds = self.loop_rounds if n_rounds is None else n_rounds
        if n_rounds < 1:
            raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
        batch = len(x)
        if depth_s is not None and tuple(depth_s.shape) != (batch, n_rounds):
            raise ValueError(f"depth_s must have shape (batch, n_rounds) = {(batch, n_rounds)}, got {tuple(depth_s.shape)}")
        # The step weights are known numbers on the grid (depth_s None) and per-sample tensors for given coordinates.
        step_weights = loop_step_weights(depth_s, n_rounds)
        if depth_s is None:  # the grid the depth embedder reads
            depth_s = torch.arange(1, n_rounds + 1, device=x[0].device, dtype=torch.float32).div_(n_rounds).expand(batch, n_rounds)

        unified, freqs, mask, adaln_input, x_size = self._front_end(x, t, cap_feats, patch_size, f_patch_size)
        for block in self.pre:
            unified = self._call(block, unified, mask, freqs, adaln_input, None, None, None)
        outputs: list[list[torch.Tensor]] = []
        if include_prelude:  # u_0: the pre blocks' output straight through the readout, no shared block
            outputs.append(self._readout(unified, mask, freqs, adaln_input, x_size, patch_size, f_patch_size))
        for i in range(n_rounds):
            u_prev = unified
            if self.loop_grad_checkpoint and torch.is_grad_enabled():  # recompute the round's block activations in the backward pass
                unified = torch.utils.checkpoint.checkpoint(self._round_blocks, unified, mask, freqs, adaln_input, depth_s, i, use_reentrant=False)
            else:
                unified = self._round_blocks(unified, mask, freqs, adaln_input, depth_s, i)
            unified = scale_round_update(u_prev, unified, step_weights[i])
            if readout == "all" or i == n_rounds - 1:
                outputs.append(self._readout(unified, mask, freqs, adaln_input, x_size, patch_size, f_patch_size))
        return outputs

    # ----------------------------------------------------------------- #
    # forward
    # ----------------------------------------------------------------- #
    def forward_rounds(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        n_rounds: int | None = None,
        readout: Literal["all", "last"] = "all",
        depth_s: torch.Tensor | None = None,
        include_prelude: bool = False,
    ) -> list[torch.Tensor]:
        """Run ``n_rounds`` rounds (default :attr:`loop_rounds`) on the latents ``hidden_states`` ``[B, C, H, W]``; return one velocity
        ``[B, C, H, W]`` per round (``readout="all"``) or the last, preceded by the prelude readout ``u_0`` with ``include_prelude=True``.
        ``timestep`` is the scheduler timestep ``sigma * 1000`` ``[B]`` (module docstring), ``encoder_hidden_states`` the zero-padded captions
        ``[B, L, D]`` and ``pooled_projections`` their validity mask ``[B, L]``; ``depth_s`` (``[B, n_rounds]``) the depth coordinates, ``None`` for
        the uniform grid. With :attr:`loop_grad_checkpoint` and gradient enabled, every round's shared block stack runs under non-reentrant
        activation checkpointing."""
        raw = self._run_rounds(
            image_list(hidden_states), model_time(timestep), caption_rows(encoder_hidden_states, pooled_projections), n_rounds, readout, depth_s,
            include_prelude,
        )
        return [velocity_batch(r) for r in raw]

    def forward(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        cap_feats: list[torch.Tensor],
        return_dict: bool = True,
        controlnet_block_samples: dict[int, torch.Tensor] | None = None,
        siglip_feats: list[list[torch.Tensor]] | None = None,
        image_noise_mask: list[list[int]] | None = None,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):
        """``ZImageTransformer2DModel.forward``'s signature: the raw output (un-negated, the teacher's list of ``[C, 1, H, W]``) of
        :attr:`loop_rounds` rounds on the grid ``s_k = k / K``, ``t`` the pipeline's ``1 - sigma``. ControlNet samples and the omni inputs (SigLIP
        features, noise masks) are not supported and raise."""
        if controlnet_block_samples is not None or siglip_feats is not None or image_noise_mask is not None or isinstance(x[0], list):
            raise NotImplementedError("LoopedZImageTransformer is the text-to-image model: no ControlNet, SigLIP or omni inputs")
        output = self._run_rounds(x, t, cap_feats, self.loop_rounds, "last", None, False, patch_size, f_patch_size)[0]
        return (output,) if not return_dict else Transformer2DModelOutput(sample=output)


class ZImageTeacherAdapter(nn.Module):
    """A frozen ``ZImageTransformer2DModel`` with the training call convention (module docstring): ``forward(hidden_states [B, C, H, W],
    timestep = sigma * 1000, encoder_hidden_states [B, L, D], pooled_projections = validity mask [B, L], return_dict=False)`` returns
    ``(velocity,)`` with the velocity ``-raw`` ``[B, C, H, W]``. ``dtype`` is the compute dtype the callers cast inputs to (the teacher's bf16)."""

    def __init__(self, transformer: ZImageTransformer2DModel, compute_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.transformer = transformer
        self._compute_dtype = compute_dtype

    @property
    def dtype(self) -> torch.dtype:
        return self._compute_dtype

    @property
    def config(self):
        return self.transformer.config

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        return_dict: bool = False,
    ):
        raw = self.transformer(image_list(hidden_states), model_time(timestep), caption_rows(encoder_hidden_states, pooled_projections), return_dict=False)[0]
        return (velocity_batch(raw),)
