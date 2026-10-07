# Registro del proyecto

Documento vivo: estado actual, hoja de ruta, decisiones y dudas abiertas.
Se actualiza en cada sesión de trabajo. Servirá de base para el informe y las diapositivas.

---

## Estado actual

**Fase actual:** F0 · Configuración del proyecto y de las dos máquinas.

**Hecho**
- Repositorio con estructura de proyecto, material del profesor (sin imágenes), entorno
  compartido, modos `rapido`/`completo`, detección de dispositivo, tests y CI.

**Siguiente**
- Portátil: clonar, crear entorno, instalar PyTorch CPU, descargar imágenes, `pytest -q`.
- PC Universidad: comprobar versión de torch instalada (¿2.14.0?), `pip install -r requirements.txt`,
  fijar `HSA_OVERRIDE_GFX_VERSION` en el entorno, `python -m src.utils.dispositivo`.
- Empezar F1 (auditoría de datos).

**Queda**
- F1 a F9 de la hoja de ruta.

---

## Hoja de ruta

Basada en el plan de trabajo del enunciado (§13) y la guía, con fases añadidas por nosotras.

| Fase | Contenido | Estado |
|---|---|---|
| F0 | Configuración: repo, entorno en las dos máquinas, descarga de datos | en curso |
| F1 | **Auditoría y visualización de datos** (añadida): pacientes, clases, cohortes, canales, rangos, ejemplos PRE/EARLY/LATE y realce, comprobación PRE < EARLY | pendiente |
| F2 | Tubería de entrenamiento: Dataset/DataLoader, bucle de entrenamiento, modos, evaluación por paciente, registro automático de cada ejecución (config, máquina, tiempos, métricas) | pendiente |
| F3 | **Diseño de la CNN (Judith)** e iteración: baselines, pérdida normal vs ponderada (`pos_weight`), sobreajuste, descartar redes que no mejoren en 5–10 épocas | pendiente |
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
