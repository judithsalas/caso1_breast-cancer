"""Las arquitecturas de Judith: AQUÍ se itera.

Cada versión es una lista de decisiones para la CNN de `src/models/cnn.py`.
Para probar una idea nueva: añadir una entrada (cnn_v2, cnn_v3...) cambiando
solo lo que se quiere comparar, y anotar en docs/REGISTRO.md qué se cambia y
por qué. No se modifican versiones ya entrenadas: si cambia algo, es otra versión.

Entrenar una versión:
    python -m src.models.comprobar cnn_v1
    python -m src.training.entrenar --modo rapido --modelo cnn_v1
"""

VERSIONES: dict[str, dict] = {
    # ---------------------------------------------------------------------- #
    # cnn_v1 · primera red (2026-10-07)
    # ---------------------------------------------------------------------- #
    "cnn_v1": {
        # 3 bloques: lo que recomienda la guía para empezar. Mapa final 32x32;
        # cada punto "ve" ~22x22 píxeles: la red detecta patrones locales de realce.
        "filtros": (16, 32, 64),
        # Kernel 3x3, estándar; padding automático para conservar el tamaño.
        "kernel": 3,
        # Una convolución por bloque: lo más simple para una primera versión.
        "convs_por_bloque": 1,
        # MaxPool: se queda con la respuesta más fuerte ("¿aparece el patrón?").
        "pooling": "max",
        # BatchNorm: entrenamiento más estable y menos sensible al learning rate.
        "batchnorm": True,
        # Pooling global: 64 valores en vez de 65.536 -> muchos menos parámetros
        # y menos sobreajuste con ~880 pacientes de entrenamiento.
        "resumen": "global",
        # Cabeza lineal directa, sin capa oculta ni dropout.
        "cabeza_oculta": None,
        "dropout": 0.0,
    },
}
