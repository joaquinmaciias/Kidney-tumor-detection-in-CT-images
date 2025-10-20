"""
Paquete de trainers de nnU-Net.

Incluye el trainer base (nnUNetTrainer) y, además, el nuevo
nnUNetTrainer_SGDR que acabamos de copiar.
"""

# ──────────────────────────────────────────────────────────────
#  trainers “oficiales”
from .nnUNetTrainer import nnUNetTrainer
# ──────────────────────────────────────────────────────────────
#  trainer SGDR (Rel-UNet)
from .nnUNetTrainer_SGDR import nnUNetTrainer_SGDR
# ──────────────────────────────────────────────────────────────
#  re-exportamos cualquier subpaquete, ej. variants
from . import variants        # deja disponibles variants.*
# ──────────────────────────────────────────────────────────────

__all__ = [
    "nnUNetTrainer",
    "nnUNetTrainer_SGDR",
    "variants",
]
