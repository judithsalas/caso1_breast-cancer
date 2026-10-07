"""CNN configurable: cada decisión de diseño es un parámetro.

Las arquitecturas concretas (cnn_v1, cnn_v2...) NO se programan aquí: se
declaran en `src/models/versiones.py` como una lista de valores para estos
parámetros. Así cada versión es reproducible y comparable con las demás.

Estructura:

    entrada (B, 3, 256, 256)       PRE, EARLY, LATE (instantes, no colores)
       │
       ▼
    EXTRACTOR: un bloque por elemento de `filtros`
       cada bloque = [conv kxk -> (BatchNorm) -> ReLU] x convs_por_bloque -> pooling 2x2
       (el pooling divide el lado entre 2: 256 -> 128 -> 64 -> ...)
       │
       ▼
    RESUMEN: "global" (media de cada canal) o "flatten" (todos los valores)
       │
       ▼
    CABEZA: [Dropout] -> [Linear -> ReLU -> Dropout] (si hay capa oculta) -> Linear -> 1 logit
"""

from __future__ import annotations

import torch
from torch import nn

LADO_ENTRADA = 256
CANALES_ENTRADA = 3


class CNN(nn.Module):
    def __init__(
        self,
        filtros: tuple[int, ...],
        kernel: int = 3,
        convs_por_bloque: int = 1,
        pooling: str = "max",
        batchnorm: bool = True,
        resumen: str = "global",
        cabeza_oculta: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        """
        filtros           canales de salida de cada bloque; su longitud = número de bloques.
        kernel            tamaño del kernel (impar). El padding se calcula para conservar el
                          tamaño del mapa: solo el pooling lo reduce.
        convs_por_bloque  convoluciones seguidas antes de cada pooling.
        pooling           "max" (¿aparece el patrón?) o "avg" (media de la zona).
        batchnorm         BatchNorm2d después de cada convolución.
        resumen           "global" (AdaptiveAvgPool: un valor por canal) o "flatten".
        cabeza_oculta     None = Linear directa a 1 salida; un número = capa intermedia.
        dropout           probabilidad de dropout en la cabeza (0 = sin dropout).
        """
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel debe ser impar para conservar el tamaño con padding")
        if pooling not in {"max", "avg"}:
            raise ValueError("pooling debe ser 'max' o 'avg'")
        if resumen not in {"global", "flatten"}:
            raise ValueError("resumen debe ser 'global' o 'flatten'")

        # ---- Extractor ------------------------------------------------------ #
        capas: list[nn.Module] = []
        canales = CANALES_ENTRADA
        for salida in filtros:
            for _ in range(convs_por_bloque):
                # Con BatchNorm detrás, el sesgo de la convolución sobra (BN ya lo aporta).
                capas.append(nn.Conv2d(canales, salida, kernel, padding=kernel // 2, bias=not batchnorm))
                if batchnorm:
                    capas.append(nn.BatchNorm2d(salida))
                capas.append(nn.ReLU(inplace=True))
                canales = salida
            capas.append(nn.MaxPool2d(2) if pooling == "max" else nn.AvgPool2d(2))
        self.extractor = nn.Sequential(*capas)

        # ---- Resumen: de mapas a vector ------------------------------------ #
        lado_final = LADO_ENTRADA // 2 ** len(filtros)
        if resumen == "global":
            self.resumen = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            n_vector = canales
        else:
            self.resumen = nn.Flatten()
            n_vector = canales * lado_final * lado_final

        # ---- Cabeza: de vector a 1 logit (sin sigmoide) --------------------- #
        cabeza: list[nn.Module] = []
        if cabeza_oculta:
            cabeza += [nn.Linear(n_vector, cabeza_oculta), nn.ReLU(inplace=True)]
            n_vector = cabeza_oculta
        if dropout > 0:
            cabeza.append(nn.Dropout(dropout))
        cabeza.append(nn.Linear(n_vector, 1))
        self.cabeza = nn.Sequential(*cabeza)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.resumen(self.extractor(x)))
