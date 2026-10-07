# CLAUDE.md

Contexto y reglas para Claude en este repositorio. **Leer al empezar cada sesión**, junto con
[`docs/REGISTRO.md`](docs/REGISTRO.md) (decisiones, hoja de ruta y estado actual).

## El proyecto

Caso BreastDCEDL: CNN 2D desde cero (PyTorch) que predice pCR a partir de las tres fases
DCE-MRI (PRE, EARLY, LATE) de un corte, con evaluación **por paciente**. Entregables:
repositorio reproducible, pesos, informe, app web desplegada con URL y 5 diapositivas.

Fuentes de verdad, por orden:
1. Enunciado: `breastdcedl/documentation/caso_breastdcedl.pdf`
2. Guía: `breastdcedl/GUIA.md`
3. Presentación del profesor, artículo de Fridman et al. y diapositivas "De la GPU a GitHub"
   (no están en el repo; Judith las tiene).

## Reglas de trabajo con Judith (obligatorias)

- **La arquitectura de la CNN la diseña Judith.** Claude ayuda, explica y, cuando haya una red
  definitiva, la dibuja. No propone una arquitectura "hecha" para que la copie.
- **Ninguna decisión sin explicarla y sin su aprobación.** Antes de subir algo a GitHub, se
  explica qué se ha hecho y por qué.
- **Ser objetivo:** si algo no se puede hacer, está mal o contradice los documentos, decirlo
  con los motivos. Si hay contradicciones entre documentos o dudas, preguntar.
- Una arquitectura se descarta si en 5–10 épocas no mejora el AUC por paciente en validación.
- Toda decisión, cambio de plan o resultado relevante se anota en `docs/REGISTRO.md`.
- Las diapositivas (exactamente 5) son lo último que se hace.

## Reglas del caso que invalidan el trabajo

1. Mezclar cortes de una misma paciente entre entrenamiento y validación → usar siempre `fold`
   / `uc.particion`.
2. Usar test para ajustar algo (modelo, hiperparámetros, umbral, método de agregación).
   Test se evalúa **una sola vez**, al final.
3. Modelos preentrenados, arquitecturas de catálogo o transfer learning.
4. Publicar la validación privada del profesor o sus etiquetas.
5. Trabajo individual.

Recordatorios técnicos: los 3 canales son instantes temporales, no RGB (nada de normalización
ImageNet ni ColorJitter); aumentos geométricos al tensor completo; `shuffle=False` al evaluar;
la normalización de la app debe ser idéntica a la del entrenamiento; no reportar solo accuracy
(predecir siempre pCR=0 da 70,6 %).

## Dos máquinas

| | Portátil | PC Universidad |
|---|---|---|
| SO | Windows | Ubuntu |
| Cálculo | CPU (GPU AMD integrada/baja, no se usa) | AMD RX 6700 XT 12 GB, ROCm |
| PyTorch | torch 2.14.0 CPU | torch 2.14.0 ROCm 7.14 + `HSA_OVERRIDE_GFX_VERSION=10.3.0` |

Mismo entorno conda `cancer` (Python 3.12) y mismo `requirements.txt` en ambas. PyTorch no
está en `requirements.txt` porque el paquete depende del hardware. El código no distingue
máquinas: `src/utils/dispositivo.py` detecta GPU o CPU. Los modos `rapido`/`completo`
(`src/config.py`) indican tipo de ejecución, no máquina.

## Convenciones

- Código y documentación en español. Commits con Conventional Commits en español
  (`feat(models): ...`, `docs: ...`).
- Material del profesor en `breastdcedl/`: se usa tal cual, no se modifica (excluido de ruff).
- `breastdcedl/dataset/` nunca se versiona. De `models/` solo se versiona `models/final/`.
- Antes de subir: `ruff check .` y `pytest -q`.
- Claude trabaja en la nube sobre una copia del repo y sube a su rama de trabajo; Judith
  fusiona en `main` y hace `git pull` en sus máquinas.
