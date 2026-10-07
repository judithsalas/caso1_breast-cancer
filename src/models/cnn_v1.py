"""Primera arquitectura de Judith: cnn_v1.

PLANTILLA: el esqueleto está hecho; las decisiones de diseño son tuyas.
Cuando la tengas, compruébala con:

    python -m src.models.comprobar cnn_v1

que te enseña la forma de la salida de cada capa y el número de parámetros.

Lo que la red DEBE cumplir (lo exige el caso):
  - Entrada: tensor (B, 3, 256, 256) en [0, 1]. Los 3 canales son PRE, EARLY y
    LATE (instantes de tiempo, no colores).
  - Salida: (B, 1), un único logit SIN sigmoide (la sigmoide va dentro de
    BCEWithLogitsLoss al entrenar, y se aplica aparte al evaluar).
  - Todo desde cero con capas de torch.nn: nada de torchvision.models ni pesos
    preentrenados.

Piezas de torch.nn que te pueden servir:
  nn.Conv2d(in_channels, out_channels, kernel_size, padding=...)
  nn.BatchNorm2d(canales)        nn.ReLU()        nn.Dropout(p)
  nn.MaxPool2d(2)                nn.AvgPool2d(2)  nn.AdaptiveAvgPool2d(1)
  nn.Flatten()                   nn.Linear(entradas, salidas)
  nn.Sequential(capa1, capa2, ...)

Preguntas que tienes que responder (y justificar en docs/REGISTRO.md):
  1. ¿Cuántos bloques convolucionales? Cada bloque suele ser conv -> (norm) ->
     activación -> pooling. La guía recomienda empezar con 3 o 4.
  2. ¿Cuántos filtros en el primer bloque, y cómo crecen en los siguientes?
  3. ¿Qué tamaño de kernel y qué padding? (¿quieres conservar el tamaño del
     mapa dentro de la convolución y reducirlo solo con el pooling?)
  4. ¿Qué tamaño tiene el mapa al final de los bloques? Calcúlalo: cada pooling
     2x2 divide el lado entre 2 (256 -> 128 -> ...).
  5. ¿Cómo pasas de mapas de características a un vector? Flatten (muchos
     valores, muchos parámetros) o un pooling global (pocos valores).
  6. ¿Qué cabeza final? ¿Una capa lineal directa a 1 salida o una capa
     intermedia antes? ¿Con dropout?
  7. ¿Usas BatchNorm? ¿Por qué?
"""

from __future__ import annotations

import torch
from torch import nn

from src.models import registrar


@registrar("cnn_v1")
class CNNv1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO (Judith): define aquí las capas, por ejemplo:
        #   self.bloques = nn.Sequential(...)
        #   self.cabeza = nn.Sequential(...)
        raise NotImplementedError("cnn_v1 todavía no está diseñada: rellena __init__ y forward")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO (Judith): x es (B, 3, 256, 256). Devuelve (B, 1).
        raise NotImplementedError
