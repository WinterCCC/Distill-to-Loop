"""Joint attention of the looped student: per-stream QK-norm and soft exclusive self attention (XSA).

:class:`LoopedJointAttnProcessor` repeats the operations of diffusers 0.40.0's ``JointAttnProcessor2_0`` (the processor every
``JointTransformerBlock`` of SD3-medium uses) in the same order and adds two operations, each present only when
:func:`install_joint_attention_options` gave the ``Attention`` module its parameters:

* QK-norm: after the head split, the image stream's ``q`` / ``k`` go through ``attn.qk_norm_img`` and the text stream's ``add_q`` / ``add_k``
  through ``attn.qk_norm_txt``. Each is a :class:`QKNorm`: RMS over ``head_dim`` in fp32 (eps 1e-6) times a learnable gain ``[heads, head_dim]``
  per tensor. The attention scale is unchanged. The fp32 intermediates are recomputed in the backward pass (activation checkpointing).
* Soft XSA (the shared loop blocks only): :func:`exclusive_self_attention` with the per-head ``attn.xsa_beta`` (``[heads]``, zero at
  initialization) applied to the joint attention output of every token against its own value vector, before the output is split into the two
  streams and projected.

The processor holds no state; the parameters live on the ``Attention`` module, so they are part of the block's ``state_dict``
(``attn.qk_norm_img.q_gain`` and so on, :data:`STUDENT_ONLY_ATTN_KEYS`). A block without these options keeps diffusers' own processor.

The processor was checked against one diffusers version only (:data:`DIFFUSERS_VERSION_WRITTEN_AGAINST`), so installing it under any other
installed version is refused.
"""

from __future__ import annotations

import diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from torch.utils.checkpoint import checkpoint

DIFFUSERS_VERSION_WRITTEN_AGAINST = "0.40.0"  # the JointAttnProcessor2_0 this processor repeats

QK_NORM_EPS = 1e-6
QK_NORM_IMG = "qk_norm_img"  # Attention attribute: the image stream's QKNorm (q, k)
QK_NORM_TXT = "qk_norm_txt"  # Attention attribute: the text stream's QKNorm (add_q, add_k)
XSA_BETA = "xsa_beta"  # Attention attribute: the per-head beta of soft XSA
# Block-relative state_dict keys these options add to a JointTransformerBlock. A teacher block has none of them, so the teacher -> student
# weight transfer leaves them at their initialization.
STUDENT_ONLY_ATTN_KEYS = frozenset(
    (*(f"attn.{module}.{name}" for module in (QK_NORM_IMG, QK_NORM_TXT) for name in ("q_gain", "k_gain")), f"attn.{XSA_BETA}")
)


# --------------------------------------------------------------------------- #
# QK-norm
# --------------------------------------------------------------------------- #
def rms_normalize_heads(x: torch.Tensor, gain: torch.Tensor, eps: float = QK_NORM_EPS) -> torch.Tensor:
    """``x`` ``[B, heads, N, head_dim]`` divided by its RMS over ``head_dim`` (``eps`` inside the root) and multiplied by ``gain``
    ``[heads, head_dim]``. Computed in fp32 whatever ``x``'s dtype, returned in ``x``'s dtype."""
    xf = F.rms_norm(x.float(), (x.shape[-1],), weight=None, eps=eps)
    return (xf * gain.float().unsqueeze(1)).to(x.dtype)


def _normalize_qk(
    q: torch.Tensor, k: torch.Tensor, q_gain: torch.Tensor, k_gain: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return rms_normalize_heads(q, q_gain, eps), rms_normalize_heads(k, k_gain, eps)


class QKNorm(nn.Module):
    """RMSNorm of every head's ``q`` and ``k`` over ``head_dim`` with learnable gains ``q_gain`` and ``k_gain`` (``[num_heads, head_dim]``,
    initialized to 1). ``forward(q, k)`` takes and returns ``[B, heads, N, head_dim]`` tensors.

    ``bypass`` (set only by the gain calibration, :mod:`distill_to_loop.training.calibration`) turns the module into the identity."""

    def __init__(self, num_heads: int, head_dim: int, eps: float = QK_NORM_EPS) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.q_gain = nn.Parameter(torch.ones(num_heads, head_dim))
        self.k_gain = nn.Parameter(torch.ones(num_heads, head_dim))
        self.bypass = False

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bypass:
            return q, k
        if torch.is_grad_enabled():
            # No RNG inside, so nothing to preserve; the recomputation replays the autocast state.
            return checkpoint(_normalize_qk, q, k, self.q_gain, self.k_gain, self.eps, use_reentrant=False, preserve_rng_state=False)
        return _normalize_qk(q, k, self.q_gain, self.k_gain, self.eps)

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}, eps={self.eps}"


# --------------------------------------------------------------------------- #
# exclusive self attention
# --------------------------------------------------------------------------- #
def exclusive_self_attention(out: torch.Tensor, v: torch.Tensor, beta: torch.Tensor | None = None) -> torch.Tensor:
    """Exclusive self attention (XSA, Zhai 2026, https://arxiv.org/abs/2603.09078): remove from each token's attention output the component
    along that token's own value vector. Adapted from the Looped-DiT implementation (MIT License, see NOTICE): the projection is computed in fp32
    and cast back to ``out.dtype``. ``out`` and ``v`` are token-aligned with heads first, ``[B, heads, N, head_dim]``.

    ``beta=None`` is the exact XSA. ``beta`` (``[heads]``) is the soft variant: the removed component is scaled per head,
    ``out - beta_h (out . v_hat) v_hat``, so ``beta = 0`` is the identity. ``beta`` is not clamped.
    """
    v_hat = F.normalize(v.float(), dim=-1)
    out_f = out.float()
    coef = (out_f * v_hat).sum(dim=-1, keepdim=True)
    if beta is not None:
        coef = beta.float().view(1, -1, 1, 1) * coef
    return (out_f - coef * v_hat).to(out.dtype)


# --------------------------------------------------------------------------- #
# the processor
# --------------------------------------------------------------------------- #
class LoopedJointAttnProcessor:
    """diffusers 0.40.0's ``JointAttnProcessor2_0`` plus the per-stream QK-norm and soft XSA of the module docstring, read off ``attn``.

    Adapted from diffusers (Apache-2.0, see NOTICE)."""

    def __init__(self) -> None:
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("LoopedJointAttnProcessor requires PyTorch 2.0 (scaled_dot_product_attention)")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        qk_norm_img = getattr(attn, QK_NORM_IMG, None)
        if qk_norm_img is not None:  # the image stream's per-head QK-norm
            query, key = qk_norm_img(query, key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            qk_norm_txt = getattr(attn, QK_NORM_TXT, None)
            if qk_norm_txt is not None:  # the text stream's per-head QK-norm, gains of its own
                encoder_hidden_states_query_proj, encoder_hidden_states_key_proj = qk_norm_txt(
                    encoder_hidden_states_query_proj, encoder_hidden_states_key_proj
                )

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        xsa_beta = getattr(attn, XSA_BETA, None)
        if xsa_beta is not None:  # soft XSA on every token's output against its own value, before the split
            hidden_states = exclusive_self_attention(hidden_states, value, xsa_beta)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def installed_versions() -> str:
    """The installed torch and diffusers versions, with the diffusers version :class:`LoopedJointAttnProcessor` repeats."""
    return f"torch {torch.__version__}, diffusers {diffusers.__version__} (LoopedJointAttnProcessor written against diffusers {DIFFUSERS_VERSION_WRITTEN_AGAINST})"


def require_diffusers_version_written_against() -> None:
    """Raise ``RuntimeError`` unless the installed diffusers is :data:`DIFFUSERS_VERSION_WRITTEN_AGAINST`: another version's
    ``JointAttnProcessor2_0`` may differ from the operations :class:`LoopedJointAttnProcessor` repeats."""
    if diffusers.__version__ != DIFFUSERS_VERSION_WRITTEN_AGAINST:
        raise RuntimeError(
            f"LoopedJointAttnProcessor repeats diffusers {DIFFUSERS_VERSION_WRITTEN_AGAINST}'s JointAttnProcessor2_0, but diffusers "
            f"{diffusers.__version__} is installed: install diffusers=={DIFFUSERS_VERSION_WRITTEN_AGAINST}, or compare the processor with this "
            "version's JointAttnProcessor2_0 (tests/test_model.py::test_processor_without_options_equals_diffusers) and update "
            "DIFFUSERS_VERSION_WRITTEN_AGAINST"
        )


def install_joint_attention_options(attn: Attention, qk_norm: bool, xsa_soft: bool) -> None:
    """Give ``attn`` (a ``JointTransformerBlock``'s joint attention) the parameters of the requested options and
    :class:`LoopedJointAttnProcessor`. ``qk_norm``: two :class:`QKNorm` (``attn.qk_norm_img``, ``attn.qk_norm_txt``, gains at 1 until the
    calibration sets them). ``xsa_soft``: ``attn.xsa_beta``, zeros ``[heads]``. Neither: nothing changes (diffusers' processor, no parameter).
    Raises ``RuntimeError`` under another diffusers version and ``TypeError`` when ``attn`` does not run diffusers' ``JointAttnProcessor2_0`` or
    has no text stream."""
    if not (qk_norm or xsa_soft):
        return
    require_diffusers_version_written_against()
    if not isinstance(attn.processor, JointAttnProcessor2_0) or getattr(attn, "add_q_proj", None) is None:
        raise TypeError(f"the student options need a joint attention running diffusers' JointAttnProcessor2_0, got {type(attn.processor).__name__}")
    heads = attn.heads
    head_dim = attn.to_k.out_features // heads
    if qk_norm:
        setattr(attn, QK_NORM_IMG, QKNorm(heads, head_dim))
        setattr(attn, QK_NORM_TXT, QKNorm(heads, head_dim))
    if xsa_soft:
        setattr(attn, XSA_BETA, nn.Parameter(torch.zeros(heads)))
    attn.set_processor(LoopedJointAttnProcessor())
