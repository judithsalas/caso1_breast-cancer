"""Toda versión de la CNN debe convertir (B, 3, 256, 256) en (B, 1)."""

import pytest
import torch

from src.models import CNN, VERSIONES, contar_parametros, crear_modelo


def salida(modelo):
    modelo.eval()
    with torch.no_grad():
        return modelo(torch.rand(2, 3, 256, 256))


@pytest.mark.parametrize("nombre", sorted(VERSIONES))
def test_versiones_dan_un_logit(nombre):
    assert salida(crear_modelo(nombre)).shape == (2, 1)


@pytest.mark.parametrize("opciones", [
    {"filtros": (8, 16, 32, 64), "convs_por_bloque": 2},
    {"filtros": (16, 32), "pooling": "avg", "batchnorm": False},
    {"filtros": (16, 32, 64), "resumen": "flatten"},
    {"filtros": (16, 32, 64), "cabeza_oculta": 32, "dropout": 0.3},
    {"filtros": (16, 32, 64), "kernel": 5},
])
def test_variantes_de_la_cnn(opciones):
    assert salida(CNN(**opciones)).shape == (2, 1)


def test_cnn_v1_es_pequena():
    assert contar_parametros(crear_modelo("cnn_v1")) < 30_000


def test_opciones_invalidas():
    with pytest.raises(ValueError):
        CNN(filtros=(16,), kernel=4)
    with pytest.raises(ValueError):
        crear_modelo("no_existe")
