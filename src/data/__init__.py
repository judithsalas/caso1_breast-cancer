"""Acceso a los datos del caso.

Rutas comunes y acceso a `utils_caso.py`, el módulo del profesor que vive en
`breastdcedl/`. Se usa tal cual (partición por paciente, Dataset, agregación);
no se copia ni se modifica.

    from src.data import RAIZ_DATOS, uc
    samples = uc.cargar_samples()
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parents[2]
RAIZ_DATOS = RAIZ_REPO / "breastdcedl"

if str(RAIZ_DATOS) not in sys.path:
    sys.path.insert(0, str(RAIZ_DATOS))

import utils_caso as uc

__all__ = ["RAIZ_DATOS", "RAIZ_REPO", "imagenes_descargadas", "uc"]


def imagenes_descargadas() -> int:
    """Número de PNG presentes en breastdcedl/dataset (deben ser 38109)."""
    carpeta = RAIZ_DATOS / "dataset"
    return sum(1 for _ in carpeta.rglob("*.png")) if carpeta.exists() else 0
