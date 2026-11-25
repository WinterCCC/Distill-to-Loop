"""Dooop on Z-Image: distill Tongyi-MAI/Z-Image into a looped, parameter-shared student whose loop count K is chosen at sampling time.

Public API::

    from dooop_zimage import load_student, load_vae_and_scheduler_factory, encode_prompts, generate_images, export_student, LoopedZImageTransformer

Import it with the ``zimage/`` directory and the repository root on ``sys.path`` (the scripts in ``zimage/scripts`` set both); the package
reuses the generic parts of ``distill_to_loop``.
"""

from dooop_zimage.inference import encode_prompts, export_student, generate_images, load_student, load_vae_and_scheduler_factory
from dooop_zimage.model import LoopedZImageTransformer

__all__ = [
    "LoopedZImageTransformer",
    "encode_prompts",
    "export_student",
    "generate_images",
    "load_student",
    "load_vae_and_scheduler_factory",
]
