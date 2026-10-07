"""Arquitecturas de la CNN (diseñadas por Judith).

- `cnn.py`:       la CNN configurable (cada decisión de diseño es un parámetro).
- `versiones.py`: las arquitecturas concretas (cnn_v1, cnn_v2...). Aquí se itera.
- `comprobar.py`: python -m src.models.comprobar cnn_v1  (capas, formas y parámetros)
"""

from __future__ import annotations

from torch import nn

from src.models.cnn import CNN
from src.models.versiones import VERSIONES

__all__ = ["CNN", "VERSIONES", "contar_parametros", "crear_modelo"]


def crear_modelo(nombre: str) -> nn.Module:
    if nombre not in VERSIONES:
        raise ValueError(f"Modelo desconocido: {nombre!r}. Disponibles: {sorted(VERSIONES)}")
    return CNN(**VERSIONES[nombre])


def contar_parametros(modelo: nn.Module) -> int:
    """Parámetros entrenables (el enunciado pide indicarlos en la diapositiva 3)."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
