# Registro del proyecto

Documento vivo: estado actual, hoja de ruta, decisiones y dudas abiertas.
Se actualiza en cada sesión de trabajo. Servirá de base para el informe y las diapositivas.

---

## Estado actual

**Fase actual:** F2 (tubería de entrenamiento) hecha · F3 (primera red de Judith) en curso.
F1 (auditoría de datos) aplazada, ver D7.

**Hecho**
- Repositorio con estructura, material del profesor (sin imágenes), entorno compartido, modos
  `rapido`/`completo`, detección de dispositivo, tests y CI.
- Portátil configurado (2026-10-07): Miniconda, entorno `cancer` con Python 3.12.15, torch
  2.14.0+cpu, 5 pacientes descargadas, 11 tests OK. Entorno aislado con `PYTHONNOUSERSITE=1`.
- Tubería de entrenamiento (`src/training/entrenar.py`), plantilla de la primera red
  (`src/models/cnn_v1.py`) y comprobador de arquitecturas (`src/models/comprobar.py`). Probado
  de principio a fin con imágenes reales y una red provisional (no subida).
- **cnn_v1 diseñada** (D10): 23.761 parámetros. Probada en modo rápido en CPU: ~12 s por época
  con 500 cortes, así que una época completa (~11.000 cortes) tardaría ~5 min en CPU.

- **Primer entrenamiento completo de cnn_v1** en el PC de la universidad (R1): funciona en la GPU
  (ROCm), 27 s por época. AUC por paciente máximo 0,585; prácticamente al nivel del azar.

**Siguiente**
- PC Universidad: puesta en marcha según `docs/PUESTA_EN_MARCHA_UNI.md`.
- En clase: `rapido` con `cnn_v1` y después `completo` con pérdida normal y ponderada.
- F1 (auditoría de datos) antes de sacar conclusiones de ningún resultado.

**Queda**
- F1, F3 a F9 de la hoja de ruta.

---

## Hoja de ruta

Basada en el plan de trabajo del enunciado (§13) y la guía, con fases añadidas por nosotras.

| Fase | Contenido | Estado |
|---|---|---|
| F0 | Configuración: repo, entorno en las dos máquinas, descarga de datos | portátil hecho · uni pendiente |
| F1 | **Auditoría y visualización de datos** (añadida): pacientes, clases, cohortes, canales, rangos, ejemplos PRE/EARLY/LATE y realce, comprobación PRE < EARLY | aplazada (D7) |
| F2 | Tubería de entrenamiento: Dataset/DataLoader, bucle de entrenamiento, modos, evaluación por paciente, registro automático de cada ejecución (config, máquina, tiempos, métricas) | hecho |
| F3 | **Diseño de la CNN (Judith)** e iteración: baselines, pérdida normal vs ponderada (`pos_weight`), sobreajuste, descartar redes que no mejoren en 5–10 épocas | en curso |
| F4 | Cerrar el modelo con validación interna: validación cruzada por paciente, método de agregación y umbral (nunca con test) | pendiente |
| F5 | Evaluación final en test, **una sola vez**: métricas por paciente y matriz de confusión. Dibujo de la red. Pesos en `models/final/` | pendiente |
| F6 | Informe: decisiones, métricas, matriz de confusión, limitaciones, uso de GPU, preguntas de discusión | pendiente |
| F7 | Aplicación web desplegada con URL (tecnología a decidir), pruebas con entradas válidas e inválidas | pendiente |
| F8 | Presentación: **exactamente 5 diapositivas** (lo último) | pendiente |
| F9 | Defensa: 5 muestras privadas en la app | pendiente |

Requisitos de la app (enunciado §9, guía C6), para no olvidarlos en F7: cargar las tres fases
de un corte o elegir un ejemplo; mostrar PRE, EARLY, LATE y el mapa de realce; validar formato
(PNG 256×256 en grises, fases ausentes o duplicadas, ficheros corruptos, valores no finitos);
mostrar probabilidad, clase, umbral, versión/checksum del modelo, dispositivo, tiempo de
respuesta, parámetros de entrenamiento y métricas internas; aviso de uso educativo sin validez
clínica; `model.eval()` sin gradientes; sin rutas arbitrarias ni ejecución de contenido subido;
límite de tamaño y formato; normalización idéntica a la del entrenamiento.

---

## Decisiones

Formato: qué se decide, por qué y alternativas descartadas.

### D1 · Un entorno conda común, PyTorch instalado aparte en cada máquina (2026-10-07)
- `environment.yml` (Python 3.12) y `requirements.txt` con versiones fijas, compartidos.
- PyTorch 2.14.0 / torchvision 0.29.0 en ambas máquinas: variante CPU en el portátil, ROCm 7.14
  en la universidad.
- **Por qué:** un entorno conda no se puede copiar entre sistemas operativos; lo que sí se
  comparte es la receta. PyTorch tiene un paquete distinto por hardware, así que no puede ir en
  un `requirements.txt` común.
- **Descartado:** el `requirements.txt` de la plantilla (es un `pip freeze` de Linux con
  paquetes `nvidia-*` que fallan en Windows e incluye librerías que no usamos).

### D2 · El portátil trabaja en CPU (2026-10-07)
- **Por qué:** su GPU AMD es de gama baja; ROCm en Windows no la soporta y DirectML obligaría a
  otra versión de PyTorch distinta de la de la universidad.
- Los entrenamientos reales se hacen en el PC de la universidad. GPU en la nube (Colab) queda
  como plan B, a valorar en F3 según los tiempos reales.

### D3 · Modos de ejecución `rapido` y `completo` (2026-10-07)
- El modo indica el tipo de ejecución, no la máquina; el dispositivo se detecta solo.
- `rapido`: 40 pacientes, 2 épocas, batch 16; sus métricas no cuentan.
- `completo`: todas las pacientes, hasta 60 épocas, parada si el AUC por paciente no mejora en
  10, **batch fijo 32** para que todos los entrenamientos sean comparables.
- Valores provisionales: se ajustarán al medir tiempos y memoria reales.

### D4 · Estructura del repositorio a partir de la plantilla UAX (2026-10-07)
- Se toma de la plantilla: carpetas `src/`, `tests/`, `notebooks/`, `models/`, `docs/`,
  configuración versionada, CI con ruff + pytest y commits convencionales.
- `CLAUDE.md` (contexto para Claude) separado de este registro.

### D5 · Se versiona el material del profesor, no las imágenes (2026-10-07)
- En Git: guía, scripts, metadatos, enunciado y figuras de `breastdcedl/` (~9 MB).
- Fuera de Git: `breastdcedl/dataset/` (38.109 PNG). Se descarga con `descargar_datos.py`, que
  salta los ficheros que ya existen.
- Ventaja extra: los tests de metadatos (sin solape de pacientes, etiqueta constante) corren
  en GitHub Actions sin necesitar las imágenes.

### D6 · Solo se versionan los pesos del modelo final (2026-10-07)
- `models/final/*.pt` se sube a GitHub (pocos MB); el resto de checkpoints, no.

### D7 · Aplazar la auditoría de datos (F1) para probar una primera red en clase (2026-10-07)
- Se adelanta la tubería de entrenamiento (F2) y la primera red (F3) para aprovechar la GPU de
  la clase.
- **Condición:** F1 se hace antes de sacar conclusiones de cualquier resultado. Los primeros
  entrenamientos sirven para comprobar que todo funciona y como referencia, no para decidir.

### D8 · Decisiones provisionales de la tubería de entrenamiento (2026-10-07)
Pendientes de revisar por Judith; cada una se puede cambiar.
- **Entrada:** el tensor (3, 256, 256) en [0, 1] tal como lo da `utils_caso` (PNG / 255), sin
  más normalización. La app deberá usar exactamente lo mismo.
- **Sin aumento de datos** por ahora: es una decisión a tomar en F3 (siempre sobre el tensor
  completo, nunca canal a canal).
- **Optimizador Adam**, lr 1e-3 (valor por defecto razonable para empezar).
- **Métrica de selección y de parada:** AUC por paciente en validación, porque no depende del
  umbral y no la engaña el desbalance como la accuracy.
- **Agregación por paciente:** media de probabilidades; **umbral 0,5**. Ambos provisionales: se
  eligen en F4 con validación interna.
- **Pérdida:** normal por defecto; `--ponderada` usa `pos_weight = N0/N1` calculado sobre las
  filas de entrenamiento (2,40 con todos los datos del fold 0).
- **Modo rápido:** usa solo pacientes ya descargadas, para poder probar con una descarga parcial.
  Sus métricas no significan nada (p. ej. 10 pacientes de validación con 1 pCR).

### D9 · Entorno aislado de paquetes externos (2026-10-07)
- En el portátil, pip encontró paquetes en la carpeta de usuario de Python
  (`AppData\Roaming\Python`) y no los instaló en el entorno. Se fija `PYTHONNOUSERSITE=1` en el
  entorno para que solo use lo suyo y sea reproducible. Igual en la universidad.
- En el portátil se instaló Miniconda (no Miniforge); los dos sirven igual. Al crear el entorno
  hubo que aceptar los términos de uso de los canales de Anaconda.

### D10 · Primera arquitectura: cnn_v1 (2026-10-07)
Diseñada por Judith. Para poder iterar, la CNN es configurable (`src/models/cnn.py`) y cada
versión es una lista de decisiones en `src/models/versiones.py`.

| Decisión | cnn_v1 | Por qué |
|---|---|---|
| Bloques | 3 | La guía recomienda empezar con 3–4; red simple |
| Filtros | 16 → 32 → 64 | Pocos: red pequeña para pocos datos (~880 pacientes); término medio entre 8→16→32 y 32→64→128 |
| Kernel / padding / convs | 3×3, padding 1, 1 conv por bloque | Estándar y lo más simple; solo el pooling reduce el tamaño |
| Pooling | MaxPool 2×2 | Se queda con la respuesta más fuerte de cada zona |
| BatchNorm | Sí, tras cada conv (sin sesgo en la conv) | Entrenamiento más estable |
| De mapas a vector | Pooling global | 64 valores en vez de 65.536: muchos menos parámetros y menos sobreajuste |
| Cabeza | Linear(64 → 1), sin dropout | Con pooling global la cabeza ya es mínima (65 parámetros) |
| **Total** | **23.761 parámetros** | |

Tamaños: 3@256² → 16@128² → 32@64² → 64@32² → 64 → 1.
Diagrama: [`docs/figuras/cnn_v1.svg`](figuras/cnn_v1.svg) (generado con `python -m src.models.dibujar cnn_v1`).
Consecuencia a vigilar: con 3 bloques cada punto del mapa final ve ~22×22 píxeles de 256×256;
la red detecta patrones locales de realce, no la forma global del tumor. Si se queda corta, una
versión con 4 bloques es la comparación natural.

---

## Resultados

Métricas por paciente en validación (fold 0: 219 pacientes, 64 con pCR), agregación por media,
umbral 0,5. Los detalles de cada ejecución están en su `runs/<ejecución>/resultados.json`
(no versionado).

### R1 · cnn_v1, pérdida normal (2026-10-07, PC Universidad)
- Ejecución `20261007-211625_cnn_v1_completo`. RX 6700 XT con ROCm 7.14, batch 32, 27 s por
  época (la primera 33 s). Paró en la época 20 (paciencia 10).
- **Mejor AUC 0,585 (época 10)**; el resto de épocas, entre 0,51 y 0,58.
- Sensibilidad casi siempre 0 con umbral 0,5: la red predice "no pCR" para casi todas.
- Pérdida de entrenamiento: de 0,603 a 0,524. La de validación sube desde la época 3 (de 0,60
  a 0,65–0,81): empieza a memorizar el entrenamiento sin generalizar.
- Referencia: 0,60 es la pérdida de predecir siempre la proporción de pCR (29 %). La red
  empieza exactamente ahí y apenas aprende.
- **Lectura:** cnn_v1 no encuentra señal útil (AUC ≈ azar). Con 219 pacientes, diferencias de
  AUC de ±0,04 son ruido, así que el "mejor" 0,585 no es distinto de las demás épocas.

---

## Dudas y contradicciones abiertas

1. **Licencia de los datos.** `breastdcedl/LICENSE` (bucket del profesor) dice **CC BY 4.0**;
   la guía y el enunciado dicen **CC BY-NC 4.0**. Hasta aclararlo, aplicamos la más restrictiva
   (CC BY-NC 4.0: sin uso comercial). Conviene preguntar al profesor.
2. **`ACLARACION_CORTES.md` no existe en el bucket** aunque la guía la cita. Su contenido
   esencial está en la guía (sección B4): no usar `mask_start`/`mask_end` para filtrar cortes en
   I-SPY.
3. **Dos versiones de `GUIA.md`.** La del bucket es más antigua; la que tiene Judith (la que está
   en el repo) coincide con los recuentos por fold de `samples.csv`. Se usa la de Judith.
4. **Versión de torch en la universidad:** pendiente de confirmar que es 2.14.0 (podría ser
   2.14.1 si se instaló la última).
5. **Licencia de nuestro código** (p. ej. Apache 2.0 como la plantilla, o MIT): pendiente de
   decidir antes de publicar.
