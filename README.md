# Predicción de pCR con DCE-MRI · Caso BreastDCEDL

CNN 2D construida desde cero con PyTorch para estimar, a partir de una resonancia
de mama con contraste tomada **antes** de la quimioterapia, la probabilidad de
respuesta patológica completa (pCR). Trabajo individual de la asignatura de
Aprendizaje Automático (UAX).

> Uso exclusivamente educativo. **No tiene validez clínica** ni es un dispositivo médico.

- Enunciado oficial: [`breastdcedl/documentation/caso_breastdcedl.pdf`](breastdcedl/documentation/caso_breastdcedl.pdf)
- Guía del caso: [`breastdcedl/GUIA.md`](breastdcedl/GUIA.md)
- Decisiones, hoja de ruta y estado del trabajo: [`docs/REGISTRO.md`](docs/REGISTRO.md)

## Estructura

```
├── breastdcedl/      material del profesor (guía, utils_caso.py, metadatos, enunciado)
│   └── dataset/      imágenes PNG: NO se versionan, se descargan (ver abajo)
├── src/
│   ├── config.py     modos de ejecución `rapido` y `completo`
│   ├── data/         acceso a los datos y a utils_caso.py
│   ├── models/       arquitecturas de la CNN
│   ├── training/     entrenamiento y evaluación por paciente
│   └── utils/        dispositivo (CPU / ROCm / CUDA), semilla
├── notebooks/        auditoría y visualización de datos
├── tests/            tests que corren en CPU (también en GitHub Actions)
├── models/final/     pesos del modelo final (lo único de models/ que se versiona)
├── docs/             registro de decisiones e informe
├── environment.yml   entorno conda (Python 3.12)
└── requirements.txt  librerías comunes con versión fija (sin PyTorch)
```

## Puesta en marcha

El proyecto se trabaja desde dos máquinas con el **mismo entorno**: Python 3.12,
las mismas librerías y la misma versión de PyTorch. Solo cambia la variante de
PyTorch según el hardware.

| | Portátil | PC Universidad |
|---|---|---|
| Sistema | Windows | Ubuntu |
| GPU | ninguna útil → **CPU** | AMD Radeon RX 6700 XT 12 GB → **ROCm** |
| Uso habitual | código, datos, modo `rapido`, app | modo `completo` (entrenamientos reales) |

### 1. Clonar

En Windows, en una ruta corta y fuera de OneDrive (límite de 260 caracteres en rutas):

```powershell
git clone https://github.com/judithsalas/caso1_breast-cancer.git C:\caso1
```

### 2. Entorno conda

**Primera vez en una máquina:**

```bash
conda env create -f environment.yml     # crea "cancer" con Python 3.12 + requirements.txt
conda activate cancer
```

Si el entorno `cancer` ya existe (PC Universidad):

```bash
conda activate cancer
pip install -r requirements.txt
```

### 3. Instalar PyTorch (distinto en cada máquina, misma versión)

**Portátil (CPU):**

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
```

**PC Universidad (ROCm):**

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/rocm7.14
# La RX 6700 XT (gfx1031) se hace pasar por gfx1030. Se guarda en el entorno una sola vez:
conda env config vars set HSA_OVERRIDE_GFX_VERSION=10.3.0
conda deactivate && conda activate cancer
```

Comprobar la instalación:

```bash
python -m src.utils.dispositivo
```

### 4. Descargar las imágenes

El material del profesor ya está en el repo. Esto solo descarga lo que falta (las ~38.000 PNG):

```bash
python breastdcedl/descargar_datos.py --destino breastdcedl
```

Debe terminar con `COMPLETO: 38109 imagenes`. Si se corta, se vuelve a lanzar y continúa.

### 5. Comprobar que todo funciona

```bash
ruff check .
pytest -q
```

## Al cambiar de ordenador

```bash
git pull
conda activate cancer
pip install -r requirements.txt     # instala solo lo que haya cambiado
```

Y antes de irse: `git add`, `git commit` y **`git push`**. Lo que no está en GitHub no existe en la otra máquina.

## Modos de ejecución

| | `rapido` | `completo` |
|---|---|---|
| Para qué | comprobar que el código funciona | entrenamiento real |
| Datos | 40 pacientes | todas las de train |
| Épocas | 2 | hasta 60, para si no mejora en 10 |
| Batch | 16 | 32 (fijo para poder comparar) |
| ¿Sus métricas cuentan? | **no** | sí |

Valores provisionales, definidos en [`src/config.py`](src/config.py).

## Licencias y atribución

Los datos derivan de **BreastDCEDL** (Fridman et al., *Scientific Data*, 2026,
[doi:10.1038/s41597-026-06589-6](https://doi.org/10.1038/s41597-026-06589-6)).
Uso docente y de investigación; ver [`breastdcedl/LICENSE`](breastdcedl/LICENSE) y la sección
de ética de la guía.
