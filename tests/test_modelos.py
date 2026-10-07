"""Toda arquitectura registrada debe convertir (B, 3, 256, 256) en (B, 1).

Las que aún son plantilla (lanzan NotImplementedError) se saltan.
"""

import pytest
import torch

from src.models import MODELOS, crear_modelo


@pytest.mark.parametrize("nombre", sorted(MODELOS))
def test_forma_de_salida(nombre):
    try:
        modelo = crear_modelo(nombre)
    except NotImplementedError:
        pytest.skip(f"{nombre} todavía es una plantilla")
    modelo.eval()
    with torch.no_grad():
        salida = modelo(torch.rand(2, 3, 256, 256))
    assert salida.shape == (2, 1)


def test_modelo_desconocido():
    with pytest.raises(ValueError):
        crear_modelo("no_existe")
