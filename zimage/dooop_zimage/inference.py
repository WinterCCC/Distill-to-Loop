"""Export, loading and sampling of the Z-Image student.

An exported student is a directory in the diffusers layout: ``config.json`` (the student's architecture) and
``diffusion_pytorch_model.safetensors`` (the EMA weights in fp32, with the teacher model id, its revision and the training step in the file's
metadata). It loads without the teacher transformer (:func:`load_student`) and samples inside a ``ZImagePipeline`` built from the teacher's
VAE and scheduler (:func:`load_vae_and_scheduler_factory`), with prompt encodings of the teacher's Qwen3 text encoder (:func:`encode_prompts`).
The loop count ``K`` is the student's ``loop_rounds``.

Sampling follows the evaluation protocol of the trained student: ``ZImagePipeline``'s own sigma grid (shift 6) with ``steps`` steps, CFG with the
empty prompt as the negative, the initial noise of image ``(prompt p, sample s)`` a function of ``(seed, p, s)`` only
(:func:`distill_to_loop.seeding.image_noise`), the prompts encoded once in chunks of :data:`ENCODE_CHUNK` with the empty prompt last, the captions
in bf16, the VAE in bf16, and on CUDA the whole pipeline call (the fp32 student included) under bf16 autocast.

Guidance convention. ``guidance_scale`` is ``ZImagePipeline``'s g, which combines ``pos + g (pos - neg)``: the usual CFG weight is
``w = g + 1`` (the default g = 8.0 is w = 9). ``g = 0`` runs the conditional branch only.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from distill_to_loop.inference import METADATA_KEY, WEIGHTS_NAME, read_student_metadata, write_generation_record  # noqa: F401 (re-exported)
from distill_to_loop.seeding import image_noise
from distill_to_loop.training.checkpoint import EMA_CHECKPOINT_KEY, read_checkpoint_payload
from dooop_zimage.model import TEACHER_CONFIG_FIELDS, LoopedZImageTransformer
from dooop_zimage.teacher import CHECKPOINT_IDENTITY_KEY, TEACHER_MODEL_ID, TEACHER_REVISION, teacher_dir
from dooop_zimage.text import ZImagePromptEncoder

STANDARD_METADATA_KEYS = ("teacher_model_id", "teacher_revision", "training_step", "ema_decay", "weights")
# Architecture options a training checkpoint may record in its ``model`` section, each with the only value this code supports; export refuses
# a checkpoint that records another value (an option the checkpoint does not record is accepted).
FIXED_MODEL_FIELDS: dict[str, Any] = {
    "family": "zimage", "loop_init": "stride", "untie_loop": False, "loop_index_cond": False, "guidance_embed": False, "depth_embed": True,
    "depth_cond": True, "loop_step_scale": "delta_s_unit", "qk_norm": "none", "loop_attn_mod": "none",
}
DEFAULT_LOOP_ROUNDS, DEFAULT_STEPS, DEFAULT_GUIDANCE_SCALE, DEFAULT_RESOLUTION = 4, 28, 8.0, 512
SAMPLING_DTYPE = torch.bfloat16  # the captions, the VAE and the autocast around the pipeline call on CUDA
ENCODE_CHUNK = 16  # prompts per text-encoder call (a prompt's bf16 encoding can change in the last bits with the batch it is encoded in)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def check_recorded_model(model_cfg: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when a checkpoint's recorded ``model`` section sets one of :data:`FIXED_MODEL_FIELDS` to another value."""
    wrong = {k: (model_cfg[k], v) for k, v in FIXED_MODEL_FIELDS.items() if k in model_cfg and model_cfg[k] != v}
    if wrong:
        raise ValueError(f"the checkpoint's student differs from the released architecture (recorded, expected): {wrong}")


def teacher_transformer_config(revision: str, hf_cache: str | None = None) -> dict[str, Any]:
    """The teacher transformer's constructor arguments (``transformer/config.json`` of the teacher snapshot ``revision``, the constructor
    defaults filled in for absent keys)."""
    from diffusers import ZImageTransformer2DModel

    stored = dict(ZImageTransformer2DModel.load_config(str(teacher_dir(revision, hf_cache)), subfolder="transformer", local_files_only=True))
    defaults = {name: p.default for name, p in inspect.signature(ZImageTransformer2DModel.__init__).parameters.items() if p.default is not inspect.Parameter.empty}
    merged = {**defaults, **stored}
    return {name: merged[name] for name in TEACHER_CONFIG_FIELDS}


def export_student(checkpoint: str | Path, out_dir: str | Path, hf_cache: str | None = None) -> Path:
    """Write the EMA weights of a training checkpoint as an exported student into ``out_dir`` and return the weight file. The checkpoint is
    memory-mapped and never written. The architecture is the recorded ``model`` section's split on the config of the teacher revision the
    checkpoint records (``extra["zimage"]``, read from the local Hugging Face cache); the weights are loaded into it with ``strict=True`` before
    they are written."""
    payload = read_checkpoint_payload(checkpoint, map_location="cpu", mmap=True)
    extra = payload.get("extra", {})
    if EMA_CHECKPOINT_KEY not in extra:
        raise ValueError(f"{checkpoint}: the checkpoint records no EMA weights")
    if CHECKPOINT_IDENTITY_KEY not in extra:
        raise ValueError(f"{checkpoint}: the checkpoint records no Z-Image identity (extra[{CHECKPOINT_IDENTITY_KEY!r}]); is it a Z-Image student checkpoint?")
    model_cfg = dict(extra["config"]["model"])
    check_recorded_model(model_cfg)
    revision = str(extra[CHECKPOINT_IDENTITY_KEY]["teacher_revision"])
    arch = {**teacher_transformer_config(revision, hf_cache), **{k: int(model_cfg[k]) for k in ("n_pre", "n_loop", "n_post", "n_rounds")}}
    state = {k: v.detach().to(torch.float32).contiguous() for k, v in extra[EMA_CHECKPOINT_KEY]["state_dict"].items()}
    with torch.device("meta"):
        student = LoopedZImageTransformer(**arch)
    student.load_state_dict(state, strict=True, assign=True)  # the strict key and shape check
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    student.save_config(out_dir)
    metadata = {
        "teacher_model_id": TEACHER_MODEL_ID, "teacher_revision": revision, "training_step": str(int(payload["step"])),
        "ema_decay": str(extra[EMA_CHECKPOINT_KEY]["decay"]), "weights": "ema",
    }
    path = out_dir / WEIGHTS_NAME
    # One metadata entry holding sorted JSON: the file header (and so the file's bytes) is then the same on every export of a checkpoint.
    save_file(state, str(path), metadata={METADATA_KEY: json.dumps(metadata, sort_keys=True)})
    return path


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_student(student_dir: str | Path, device: str | torch.device = "cpu", loop_rounds: int = DEFAULT_LOOP_ROUNDS) -> LoopedZImageTransformer:
    """An exported student in fp32 on ``device``, in eval mode, running ``loop_rounds`` rounds (``K``; change it later with
    ``student.loop_rounds = K``). The weights are loaded with ``strict=True``."""
    student_dir = Path(student_dir)
    config = LoopedZImageTransformer.load_config(student_dir)
    with torch.device("meta"):
        student = LoopedZImageTransformer.from_config(config)
    # Fresh storage, then a copy: tensors backed by the file's buffer are not 64-byte aligned, and CPU kernels on unaligned weights may sum in
    # another order (outputs then differ in the last bits). Every tensor of the student is in the state dict (the RoPE tables are computed on
    # first use). ``float()`` rather than ``to(dtype=...)``: diffusers' ``ModelMixin.to`` logs a warning about its list of float32 modules.
    student = student.to_empty(device=device).float()
    student.load_state_dict(load_file(str(student_dir / WEIGHTS_NAME)), strict=True)
    student.loop_rounds = int(loop_rounds)
    student.eval()
    return student


def load_vae_and_scheduler_factory(device: str | torch.device, revision: str = TEACHER_REVISION, cache_dir: str | None = None):
    """``(vae, scheduler_factory)`` of the teacher snapshot ``revision``: the VAE in bf16 (on every device, as the evaluation) and a function
    returning a fresh copy of its ``FlowMatchEulerDiscreteScheduler``."""
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

    model_dir = str(teacher_dir(revision, cache_dir))
    vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae", local_files_only=True, torch_dtype=SAMPLING_DTYPE).to(device)

    def scheduler_factory():
        return FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler", local_files_only=True)

    return vae, scheduler_factory


def build_pipeline(vae: Any, scheduler: Any, transformer: Any) -> Any:
    """A ``ZImagePipeline`` without text encoder and tokenizer (the prompt encodings are passed in)."""
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def encode_prompts(texts: Sequence[str], device: str | torch.device, revision: str = TEACHER_REVISION, cache_dir: str | None = None,
                   encoder: ZImagePromptEncoder | None = None) -> dict[str, Any]:
    """The bf16 encodings of ``texts`` and of the empty prompt (the negative), from the teacher's text encoder (loaded here and released, unless
    ``encoder`` is given), in chunks of :data:`ENCODE_CHUNK` over ``[*texts, ""]``: ``{"prompt_embeds": [len_i, 2560] per prompt,
    "negative_prompt_embeds": [len, 2560]}`` on the CPU. A prompt of more than 512 tokens raises (the pipeline would truncate it)."""
    own = encoder is None
    if own:
        encoder = ZImagePromptEncoder.from_pretrained(teacher_dir(revision, cache_dir), device, dtype=SAMPLING_DTYPE)
    texts = [*texts, ""]
    rows = [r.cpu() for i in range(0, len(texts), ENCODE_CHUNK) for r in encoder.rows(texts[i : i + ENCODE_CHUNK])]
    if own:
        del encoder
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return {"prompt_embeds": rows[:-1], "negative_prompt_embeds": rows[-1]}


def latent_shape(student: LoopedZImageTransformer, vae: Any, resolution: int) -> tuple[int, int, int]:
    """The ``(C, h, w)`` initial-noise shape of a square image of ``resolution`` pixels, which must be a multiple of the VAE's downsampling
    factor times the patch size (8 x 2 = 16 for Z-Image)."""
    factor = 2 ** (len(vae.config.block_out_channels) - 1)
    multiple = factor * int(student.config.all_patch_size[0])
    if resolution < 1 or resolution % multiple:
        raise ValueError(f"the image size must be a positive multiple of {multiple} (VAE factor {factor} x patch size), got {resolution}")
    return (int(student.config.in_channels), resolution // factor, resolution // factor)


def sampling_autocast(device: torch.device | str) -> torch.autocast:
    """bf16 autocast around a pipeline call on CUDA; disabled on any other device."""
    return torch.autocast(device_type="cuda", dtype=SAMPLING_DTYPE, enabled=torch.device(device).type == "cuda")


def generate_images(
    student: LoopedZImageTransformer,
    vae: Any,
    scheduler_factory: Any,
    encodings: Mapping[str, Any],
    *,
    loop_rounds: int = DEFAULT_LOOP_ROUNDS,
    steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    resolution: int = DEFAULT_RESOLUTION,
    seed: int = 0,
    num_samples: int = 1,
    micro_batch_size: int = 4,
    device: str | torch.device = "cuda",
    output_type: str = "pil",
) -> list[tuple[int, int, Any]]:
    """``[(prompt index, sample index, image)]`` for every prompt of ``encodings`` (:func:`encode_prompts`) and sample, prompt-major: the
    items of :func:`iter_images` with the same arguments, in one list."""
    return list(iter_images(
        student, vae, scheduler_factory, encodings, loop_rounds=loop_rounds, steps=steps, guidance_scale=guidance_scale, resolution=resolution,
        seed=seed, num_samples=num_samples, micro_batch_size=micro_batch_size, device=device, output_type=output_type,
    ))


@torch.no_grad()
def iter_images(
    student: LoopedZImageTransformer,
    vae: Any,
    scheduler_factory: Any,
    encodings: Mapping[str, Any],
    *,
    loop_rounds: int = DEFAULT_LOOP_ROUNDS,
    steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    resolution: int = DEFAULT_RESOLUTION,
    seed: int = 0,
    num_samples: int = 1,
    micro_batch_size: int = 4,
    device: str | torch.device = "cuda",
    output_type: str = "pil",
) -> Iterator[tuple[int, int, Any]]:
    """Yield ``(prompt index, sample index, image)`` for every prompt of ``encodings`` (:func:`encode_prompts`) and sample, prompt-major, as
    soon as the pipeline call that makes it returns. Image ``(p, s)`` starts from :func:`~distill_to_loop.seeding.image_noise` of
    ``(seed, p, s)``. Every pipeline call has exactly ``micro_batch_size`` rows (a short last call is padded with copies of its last row, whose
    images are dropped), so the batch shape does not depend on the number of prompts. ``guidance_scale`` is the pipeline's g (module
    docstring). Yields PIL images, or latents with ``output_type="latent"``. Like any generator, it runs (and checks ``resolution``) only when
    iterated."""
    device = torch.device(device)
    student.loop_rounds = int(loop_rounds)
    shape = latent_shape(student, vae, resolution)
    pipe = build_pipeline(vae, scheduler_factory(), student)
    negative = encodings["negative_prompt_embeds"].to(device, SAMPLING_DTYPE)
    pairs = [(p, s) for p in range(len(encodings["prompt_embeds"])) for s in range(num_samples)]
    for start in range(0, len(pairs), micro_batch_size):
        chunk = pairs[start : start + micro_batch_size]
        rows = chunk + [chunk[-1]] * (micro_batch_size - len(chunk))
        noise = torch.stack([image_noise(seed, p, s, shape) for p, s in rows])
        positives = [encodings["prompt_embeds"][p].to(device, SAMPLING_DTYPE) for p, _ in rows]
        with sampling_autocast(device):
            images = pipe(
                prompt=None, prompt_embeds=positives, negative_prompt_embeds=[negative] * len(positives), latents=noise.to(device, torch.float32),
                height=resolution, width=resolution, num_inference_steps=steps, guidance_scale=float(guidance_scale), output_type=output_type,
                return_dict=False,
            )[0]
        yield from ((p, s, image) for (p, s), image in zip(chunk, images))
