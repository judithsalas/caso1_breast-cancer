# Puesta en marcha en el PC de la universidad (una sola vez)

En el PC de la uni hay una carpeta antigua con `breastdcedl/` (incluido el dataset) que nunca
llegó a GitHub. **No hacer `git pull` sobre ella**: ahora `breastdcedl/` está en el repo y Git
se negaría a sobrescribir esos archivos. En su lugar, se clona el repo en una carpeta nueva y se
mueve el dataset ya descargado, sin volver a bajarlo.

Sustituir `RUTA_ANTIGUA` por la carpeta del proyecto antiguo.

```bash
# 0. (Opcional) Ver si la carpeta antigua tenía commits sin subir, por si hubiera algo útil
cd RUTA_ANTIGUA && git status && git log --oneline -5

# 1. Clonar en una carpeta nueva
cd ~
git clone https://github.com/judithsalas/caso1_breast-cancer.git caso1
cd ~/caso1

# 2. Mover el dataset ya descargado (no se copia: se mueve, es instantáneo)
mv RUTA_ANTIGUA/breastdcedl/dataset ~/caso1/breastdcedl/dataset

# 3. Entorno: aislarlo y fijar la variable de la GPU (se guardan dentro del entorno)
conda env config vars set PYTHONNOUSERSITE=1 HSA_OVERRIDE_GFX_VERSION=10.3.0 -n cancer
conda activate cancer
pip install -r requirements.txt

# 4. Comprobaciones
python -c "import torch; print(torch.__version__)"    # debe ser 2.14.0+rocm...
python -m src.utils.dispositivo                       # backend rocm, AMD Radeon RX 6700 XT, ~12 GB
python breastdcedl/descargar_datos.py --destino breastdcedl   # debe decir COMPLETO: 38109 (baja lo que falte)
pytest -q
```

Si `torch.__version__` no es 2.14.0, apuntarlo en `docs/REGISTRO.md` (duda abierta 4) antes de
reinstalar nada:

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/rocm7.14
```

La carpeta antigua no se borra hasta comprobar que todo funciona en `~/caso1`.

## Después, en cada sesión

```bash
cd ~/caso1
git pull
conda activate cancer
pip install -r requirements.txt
```
