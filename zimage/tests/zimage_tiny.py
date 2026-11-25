"""Tiny random Z-Image components for CPU tests: a teacher transformer of 12 layers (3 + 6 + 3, the released split), the teacher's scheduler
config, a prepared training split in the layout of ``zimage/scripts/prepare_coco.py``, and a one-rank CPU run of the real trainer on them."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

from dooop.data.latents import LatentIndex, num_shards, shard_name, shard_range, write_captions, write_index, write_meta, write_shard, write_shard_marker
from dooop.data.text import ATTN_MASK, POOLED, TEXT_EMB, shard_text_name, write_empty_prompt, write_shard_text
from dooop.training.distributed import DistInfo
from dooop_zimage.teacher import TEACHER_REVISION

TINY_TEACHER = dict(
    all_patch_size=(2,), all_f_patch_size=(1,), in_channels=16, dim=64, n_layers=12, n_refiner_layers=1, n_heads=2, n_kv_heads=2, norm_eps=1e-5,
    qk_norm=True, cap_feat_dim=16, rope_theta=256.0, t_scale=1000.0, axes_dims=[8, 12, 12], axes_lens=[128, 32, 32],
)
C, H = 16, 8  # latent channels and spatial size: 4 x 4 = 16 image tokens with the 2 x 2 patch (padded to 32)
TEXT_ROWS, D = 48, 16  # stored caption rows and the caption width
SCHEDULER_CONFIG = {"num_train_timesteps": 1000, "use_dynamic_shifting": False, "shift": 6.0}  # the teacher's scheduler_config.json
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "zimage_dooop.yaml"
PER_GPU = 24  # 16 data rows + 8 rollout rows (1 trajectory x 8 Euler steps), the smallest batch of the recipe's rollout ratio
CPU = DistInfo(rank=0, world_size=1, local_rank=0, device=torch.device("cpu"))


def tiny_transformer(seed: int = 999) -> ZImageTransformer2DModel:
    """A random fp32 tiny teacher transformer with every parameter (pad tokens and norm weights included) moved off its init. Built under a
    forked global RNG, so building it does not move the caller's random streams (as loading pretrained weights does not)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        teacher = ZImageTransformer2DModel(**TINY_TEACHER)
        with torch.no_grad():
            for p in teacher.parameters():
                p.add_(0.05 * torch.randn(p.shape))
    teacher.requires_grad_(False)
    return teacher.eval()


def tiny_teacher(device: torch.device | str = "cpu", seed: int = 999):
    """:func:`tiny_transformer` in bf16 behind the teacher adapter (compute dtype bf16), as the trainer loads the real one."""
    from dooop_zimage.model import ZImageTeacherAdapter

    return ZImageTeacherAdapter(tiny_transformer(seed).to(device=device, dtype=torch.bfloat16)).eval()


def write_tiny_split(split_dir: Path, n: int, shard_size: int = 16, seed: int = 0, text_dtype: torch.dtype = torch.float32,
                     revision: str = TEACHER_REVISION) -> Path:
    """A prepared split of ``n`` random samples (latent posteriors, captions of 1 to ``TEXT_ROWS`` rows in ``text_dtype`` with their validity
    masks, the empty prompt of 3 rows) recorded against the teacher ``revision``."""
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
        lengths = torch.randint(1, TEXT_ROWS + 1, (m,), generator=g)
        mask = torch.arange(TEXT_ROWS)[None, :] < lengths[:, None]
        text = torch.randn(m, TEXT_ROWS, D, generator=g) * mask[..., None]
        write_shard_text(split_dir / shard_text_name(k), {TEXT_EMB: text.to(text_dtype), POOLED: mask.to(text_dtype), ATTN_MASK: mask})
        write_shard_marker(split_dir, k)
        shards.append({"file": shard_name(k), "n": m})
    write_captions(split_dir, captions)
    write_meta(split_dir, ({"index": i, "prompt": c} for i, c in enumerate(captions)))
    empty_mask = torch.arange(TEXT_ROWS) < 3
    empty = torch.randn(TEXT_ROWS, D, generator=g) * empty_mask[:, None]
    write_empty_prompt(split_dir, empty.to(torch.bfloat16), empty_mask.to(torch.bfloat16), model_snapshot=revision, max_sequence_length=TEXT_ROWS)
    write_index(split_dir, LatentIndex(total=n, shard_size=shard_size, latent_shape=[C, H, H], scaling_factor=0.3611, shift_factor=0.1159,
                                       shards=shards, text_sharded=True))
    return split_dir


def tiny_vae():
    """A random fp32 ``AutoencoderKL`` with the teacher VAE's downsampling factor (8: four levels), 16 latent channels and its scaling and
    shift factors."""
    from diffusers import AutoencoderKL

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return AutoencoderKL(
            in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
            block_out_channels=(8, 8, 8, 8), layers_per_block=1, latent_channels=C, norm_num_groups=4, sample_size=64, scaling_factor=0.3611,
            shift_factor=0.1159, use_quant_conv=False, use_post_quant_conv=False,
        ).eval()


def write_tiny_hub_cache(cache_dir: Path, revision: str = TEACHER_REVISION) -> Path:
    """A Hugging Face hub cache holding a stand-in teacher snapshot ``revision``: the tiny transformer's config, the teacher's scheduler config
    and the tiny VAE (no text encoder). Returns ``cache_dir``."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    from dooop_zimage.teacher import TEACHER_MODEL_ID

    repo = cache_dir / ("models--" + TEACHER_MODEL_ID.replace("/", "--"))
    snapshot = repo / "snapshots" / revision
    tiny_transformer().save_config(str(snapshot / "transformer"))
    FlowMatchEulerDiscreteScheduler(**SCHEDULER_CONFIG).save_config(str(snapshot / "scheduler"))
    tiny_vae().save_pretrained(str(snapshot / "vae"))
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(revision, encoding="utf-8")
    return cache_dir


def tiny_encodings(n: int, seed: int = 0) -> dict[str, Any]:
    """Prompt encodings in the form of ``encode_prompts``: ``n`` bf16 captions of 1 to ``TEXT_ROWS`` rows and a 3-row negative."""
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, TEXT_ROWS + 1, (n,), generator=g).tolist()
    return {
        "prompt_embeds": [torch.randn(k, D, generator=g).to(torch.bfloat16) for k in lengths],
        "negative_prompt_embeds": torch.randn(3, D, generator=g).to(torch.bfloat16),
    }


TEXT_PIPE_WIDTH = 4  # the caption width of FakeTextPipe


class FakeTokenizer:
    """One token per word; the chat template is the prompt itself."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and enable_thinking
        return messages[0]["content"]

    def __call__(self, texts):
        class _Out:
            input_ids = [t.split() for t in texts]

        return _Out()


class FakeTextPipe:
    """A stand-in ``ZImagePipeline`` with a tokenizer and a text encoder: ``_encode_prompt`` returns ``min(words, max_sequence_length)`` rows
    (at least one) per prompt, filled with the prompt's character count, records the prompts of every call, and empties the list it is given
    (the real pipeline rewrites it in place)."""

    def __init__(self) -> None:
        self.text_encoder = torch.nn.Identity()
        self.tokenizer = FakeTokenizer()
        self.calls: list[list[str]] = []

    def _encode_prompt(self, prompt, device, max_sequence_length):
        self.calls.append(list(prompt))
        rows = [torch.full((min(max(len(p.split()), 1), max_sequence_length), TEXT_PIPE_WIDTH), float(len(p))).to(torch.bfloat16) for p in prompt]
        prompt.clear()
        return rows


def use_tiny_teacher(setattr_fn: Callable[[Any, str, Any], None]) -> Any:
    """Point the trainer at :func:`tiny_teacher` and :data:`SCHEDULER_CONFIG` (``setattr_fn``: ``monkeypatch.setattr`` or ``setattr``); returns
    the trainer module."""
    import dooop_zimage.trainer as trainer

    setattr_fn(trainer, "load_teacher", lambda cfg, device: tiny_teacher(device))
    setattr_fn(trainer, "load_scheduler_config", lambda cfg: dict(SCHEDULER_CONFIG))
    return trainer


def tiny_config(output_dir: Path, split: Path, max_steps: int, name: str = "release", overrides: Sequence[str] = ()):
    """The released config with ``PER_GPU`` rows, ``data.num_workers=0`` and a checkpoint every step, plus ``overrides``."""
    from dooop_zimage.config import load_config

    return load_config(CONFIG, [
        f"name={name}", f"output_dir={output_dir}", f"data.train_dir={split}", "data.num_workers=0", f"train.per_gpu_batch_size={PER_GPU}",
        f"train.max_steps={max_steps}", "train.checkpoint_every=1", "train.log_every=1", *overrides,
    ])


def run_training(output_dir: Path, split: Path, max_steps: int, monkeypatch: Any, name: str = "release", resume_from: Path | None = None,
                 overrides: Sequence[str] = ()) -> Path:
    """The trainer's ``run_training`` on one CPU rank with the tiny teacher (:func:`tiny_config`); returns the run directory."""
    trainer = use_tiny_teacher(monkeypatch.setattr)
    cfg = tiny_config(output_dir, split, max_steps, name, overrides)
    trainer.run_training(cfg, CPU, resume_from=resume_from)
    return cfg.run_dir
