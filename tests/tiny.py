"""Tiny random components for CPU tests: a teacher transformer, a prepared training split in the layout of ``scripts/prepare_coco.py``, and a
one-rank CPU run of the real trainer on them."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

from distill_to_loop.data.latents import LatentIndex, num_shards, shard_name, shard_range, write_captions, write_index, write_meta, write_shard, write_shard_marker
from distill_to_loop.data.text import ATTN_MASK, POOLED, TEXT_EMB, shard_text_name, write_empty_prompt, write_shard_text

C, H = 16, 8  # latent channels and spatial size
SEQ, D, D_POOLED = 6, 8, 12  # text sequence length and widths
SNAPSHOT = "tiny-snapshot"
RELEASE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "sd3m_dooop.yaml"
PER_GPU = 24  # 16 data rows + 8 rollout rows (1 trajectory x 8 Euler steps), the smallest batch of the recipe's rollout ratio


def tiny_teacher(device: torch.device | str = "cpu", num_layers: int = 9, seed: int = 999) -> SD3Transformer2DModel:
    """A random bf16 SD3 transformer with ``num_layers`` layers (``(num_layers - 4) % 5 == 0`` for a 2 + 5 + 2 student). Built under a forked
    global RNG, so building it does not move the caller's random streams (as loading pretrained weights does not)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        teacher = SD3Transformer2DModel(
            sample_size=H, patch_size=2, in_channels=C, out_channels=C, attention_head_dim=8, num_attention_heads=2, joint_attention_dim=D,
            caption_projection_dim=16, pooled_projection_dim=D_POOLED, pos_embed_max_size=H, num_layers=num_layers,
        )
    teacher = teacher.to(device=device, dtype=torch.bfloat16)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def tiny_scheduler() -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)


def write_tiny_split(split_dir: Path, n: int, shard_size: int = 16, seed: int = 0, text_dtype: torch.dtype = torch.float32) -> Path:
    """A prepared split of ``n`` random samples (latent posteriors, text encodings in ``text_dtype``, the empty prompt) recorded against
    :data:`SNAPSHOT`. The real split stores bf16 encodings; the trainer casts them to fp32 on the CPU (no bf16 autocast there), and fp32 here
    skips that cast."""
    g = torch.Generator().manual_seed(seed)
    split_dir.mkdir(parents=True, exist_ok=True)
    captions = [f"caption {i}" for i in range(n)]
    shards = []
    for k in range(num_shards(n, shard_size)):
        rows = shard_range(k, n, shard_size)
        m = len(rows)
        mean = torch.randn(m, C, H, H, generator=g)
        std = 0.1 * torch.rand(m, C, H, H, generator=g)
        write_shard(split_dir / shard_name(k), mean, std, torch.arange(rows.start, rows.stop))
        text = {
            TEXT_EMB: torch.randn(m, SEQ, D, generator=g).to(text_dtype),
            POOLED: torch.randn(m, D_POOLED, generator=g).to(text_dtype),
            ATTN_MASK: torch.ones(m, SEQ, dtype=torch.bool),
        }
        write_shard_text(split_dir / shard_text_name(k), text)
        write_shard_marker(split_dir, k)
        shards.append({"file": shard_name(k), "n": m})
    write_captions(split_dir, captions)
    write_meta(split_dir, ({"index": i, "prompt": c} for i, c in enumerate(captions)))
    write_empty_prompt(
        split_dir, torch.randn(SEQ, D, generator=g).to(text_dtype), torch.randn(D_POOLED, generator=g).to(text_dtype),
        model_snapshot=SNAPSHOT, max_sequence_length=SEQ // 2,
    )
    write_index(split_dir, LatentIndex(total=n, shard_size=shard_size, latent_shape=[C, H, H], scaling_factor=1.5305, shift_factor=0.0609, shards=shards, text_sharded=True))
    return split_dir


def run_training(output_dir: Path, split: Path, max_steps: int, monkeypatch: Any, name: str = "release", resume_from: Path | None = None,
                 overrides: Sequence[str] = ()) -> Path:
    """The release trainer's ``run_training`` on one CPU rank with :func:`tiny_teacher` and :func:`tiny_scheduler` (the released config with
    ``PER_GPU`` rows, ``data.num_workers=0`` and a checkpoint every step, plus ``overrides``); returns the run directory."""
    import distill_to_loop.training.trainer as trainer
    from distill_to_loop.config import load_config
    from distill_to_loop.training.distributed import DistInfo

    monkeypatch.setattr(trainer, "load_teacher", lambda cfg, device: tiny_teacher(device))
    monkeypatch.setattr(trainer, "load_scheduler", lambda cfg: tiny_scheduler())
    monkeypatch.setattr(trainer, "teacher_snapshot", lambda cfg: SNAPSHOT)
    cfg = load_config(RELEASE_CONFIG, [
        f"name={name}", f"output_dir={output_dir}", f"data.train_dir={split}", "data.num_workers=0", f"train.per_gpu_batch_size={PER_GPU}",
        f"train.max_steps={max_steps}", "train.checkpoint_every=1", "train.log_every=1", *overrides,
    ])
    trainer.run_training(cfg, DistInfo(rank=0, world_size=1, local_rank=0, device=torch.device("cpu")), resume_from=resume_from)
    return cfg.run_dir


def read_payload(run_dir: Path, step: int) -> dict:
    return torch.load(Path(run_dir) / f"step_{step:08d}.pt", map_location="cpu", weights_only=False)


def assert_training_states_equal(a: dict, b: dict) -> None:
    """Two checkpoint payloads hold bit-identical weights, EMA, AdamW state, scheduler state and data position."""
    assert list(a["model"]) == list(b["model"])
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
    ema_a, ema_b = a["extra"]["ema"]["state_dict"], b["extra"]["ema"]["state_dict"]
    assert list(ema_a) == list(ema_b) and all(torch.equal(ema_a[k], ema_b[k]) for k in ema_a)
    sa, sb = a["optimizer"]["state"], b["optimizer"]["state"]
    assert sorted(sa) == sorted(sb)
    for i in sa:
        for name in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(sa[i][name], sb[i][name]), (i, name)
    assert a["scheduler"] == b["scheduler"]
    assert a["extra"]["data_progress"] == b["extra"]["data_progress"]
