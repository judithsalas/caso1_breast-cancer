"""Arquitecturas de la CNN (diseñadas por Judith).

Cada arquitectura vive en su propio fichero (cnn_v1.py, cnn_v2.py...) y se
registra con un nombre para poder elegirla al entrenar:

    python -m src.training.entrenar --modo rapido --modelo cnn_v1

Para añadir una red nueva: copiar cnn_v1.py a cnn_vN.py, cambiar el nombre del
registro y añadir su import al final de este fichero.
"""

from __future__ import annotations

from torch import nn

MODELOS: dict[str, type[nn.Module]] = {}


def registrar(nombre: str):
    """Decorador que da de alta una arquitectura con un nombre."""

    def decorador(clase: type[nn.Module]) -> type[nn.Module]:
        if nombre in MODELOS:
            raise ValueError(f"Ya hay un modelo registrado como {nombre!r}")
        MODELOS[nombre] = clase
        return clase

    return decorador


def crear_modelo(nombre: str) -> nn.Module:
    if nombre not in MODELOS:
        raise ValueError(f"Modelo desconocido: {nombre!r}. Disponibles: {sorted(MODELOS)}")
    return MODELOS[nombre]()


def contar_parametros(modelo: nn.Module) -> int:
    """Parámetros entrenables (el enunciado pide indicarlos en la diapositiva 3)."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


# Registro de arquitecturas: un import por fichero.
from src.models import cnn_v1  # noqa: F401
