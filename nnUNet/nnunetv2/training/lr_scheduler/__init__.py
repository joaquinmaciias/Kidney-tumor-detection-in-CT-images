"""
Planificadores de tasa de aprendizaje para nnU-Net.

Contiene el PolyLR original y el CosineAnnealingWarmRestartsLR (SGDR).
"""

# ──────────────────────────────────────────────────────────────
from .polylr import PolyLRScheduler
from .sgdrlr import CosineAnnealingWarmRestartsLR
# ──────────────────────────────────────────────────────────────

__all__ = [
    "PolyLRScheduler",
    "CosineAnnealingWarmRestartsLR",
]
