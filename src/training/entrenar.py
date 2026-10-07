"""Entrenamiento de una arquitectura con evaluación por paciente.

    python -m src.training.entrenar --modo rapido   --modelo cnn_v1
    python -m src.training.entrenar --modo completo --modelo cnn_v1
    python -m src.training.entrenar --modo completo --modelo cnn_v1 --ponderada   # pos_weight = N0/N1

Cada ejecución crea una carpeta en runs/ (no se versiona) con:
    resultados.json   configuración, máquina, métricas por época, tiempos y memoria
    historial.csv     lo mismo por época, en tabla (para hacer gráficas)
    mejor.pt          pesos de la época con mejor AUC por paciente en validación

Qué se mide: tras cada época se predice cada corte de validación, se agrega por
paciente (media de probabilidades, umbral 0,5) con uc.evaluar_por_paciente y se
guarda AUC, sensibilidad, especificidad y matriz de confusión. El criterio para
quedarse con una época y para parar es el AUC por paciente. El conjunto test no
se toca aquí.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import Config, obtener_config
from src.data import RAIZ_DATOS, RAIZ_REPO, uc
from src.models import VERSIONES, contar_parametros, crear_modelo
from src.utils.dispositivo import describir_dispositivo, elegir_dispositivo
from src.utils.semilla import fijar_semilla

CARPETA_RUNS = RAIZ_REPO / "runs"
METODO_AGREGACION = "mean"   # provisional: se elige en F4 con validación interna
UMBRAL = 0.5                 # provisional: se elige en F4 con validación interna


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #

def pacientes_con_imagenes(filas: pd.DataFrame) -> pd.DataFrame:
    """Filas de las pacientes cuyas imágenes están descargadas (todas sus fases)."""
    existe = filas.apply(
        lambda f: all((RAIZ_DATOS / f[c]).exists() for c in ("path_pre", "path_early", "path_late")),
        axis=1,
    )
    completas = existe.groupby(filas.patient_id).all()
    return filas[filas.patient_id.isin(completas[completas].index)]


def elegir_pacientes(filas: pd.DataFrame, n: int, semilla: int) -> pd.DataFrame:
    """Hasta n pacientes al azar (reproducible), con todos sus cortes."""
    ids = pd.Series(sorted(filas.patient_id.unique()))
    elegidas = ids.sample(n=min(n, len(ids)), random_state=semilla)
    return filas[filas.patient_id.isin(elegidas)]


def preparar_datos(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición por paciente (columna fold) y, en modo rápido, un subconjunto."""
    entrena, valida = uc.particion(uc.cargar_samples(), fold_val=cfg.fold_val)

    if cfg.n_pacientes is not None:
        # Solo pacientes ya descargadas, para poder probar con una descarga parcial.
        entrena = elegir_pacientes(pacientes_con_imagenes(entrena), cfg.n_pacientes, cfg.semilla)
        valida = elegir_pacientes(pacientes_con_imagenes(valida), max(cfg.n_pacientes // 4, 2), cfg.semilla)
        if entrena.empty or valida.empty:
            raise SystemExit(
                "No hay suficientes pacientes descargadas para el modo rápido "
                f"(entrena: {entrena.patient_id.nunique()}, valida: {valida.patient_id.nunique()}).\n"
                "Descarga más: python breastdcedl/descargar_datos.py --destino breastdcedl --pacientes 100"
            )
    else:
        faltan = [r for r in pd.concat([entrena, valida]).path_pre if not (RAIZ_DATOS / r).exists()]
        if faltan:
            raise SystemExit(
                f"Faltan imágenes de {len(faltan)} cortes para el modo completo.\n"
                "Descarga todo: python breastdcedl/descargar_datos.py --destino breastdcedl"
            )

    return entrena.reset_index(drop=True), valida.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Entrenamiento y evaluación
# --------------------------------------------------------------------------- #

def predecir(modelo: nn.Module, cargador: DataLoader, dispositivo: torch.device,
             criterio: nn.Module | None = None) -> tuple[np.ndarray, float]:
    """Probabilidad de pCR de cada corte, en el orden del cargador (shuffle=False)."""
    modelo.eval()
    probabilidades, perdida_total, n = [], 0.0, 0
    with torch.no_grad():
        for x, y in cargador:
            x, y = x.to(dispositivo), y.to(dispositivo)
            logits = modelo(x).squeeze(1)
            if criterio is not None:
                perdida_total += criterio(logits, y).item() * len(y)
                n += len(y)
            probabilidades.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilidades), (perdida_total / n if n else float("nan"))


def entrenar_una_epoca(modelo, cargador, criterio, optimizador, dispositivo) -> float:
    modelo.train()
    perdida_total, n = 0.0, 0
    for x, y in cargador:
        x, y = x.to(dispositivo), y.to(dispositivo)
        optimizador.zero_grad()
        perdida = criterio(modelo(x).squeeze(1), y)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item() * len(y)
        n += len(y)
    return perdida_total / n


def entrenar_modelo(modelo, cargador_tr, cargador_va, valida, criterio, optimizador,
                    dispositivo, cfg: Config, carpeta: Path, resultados: dict) -> dict:
    """Bucle de épocas con parada temprana por AUC por paciente en validación."""
    mejor_auc, sin_mejora = -math.inf, 0
    resultados["epocas"] = []
    inicio = time.perf_counter()

    for epoca in range(1, cfg.epocas_max + 1):
        t0 = time.perf_counter()
        perdida_tr = entrenar_una_epoca(modelo, cargador_tr, criterio, optimizador, dispositivo)
        probs, perdida_va = predecir(modelo, cargador_va, dispositivo, criterio)
        m = uc.evaluar_por_paciente(probs, valida, umbral=UMBRAL, metodo=METODO_AGREGACION)
        segundos = time.perf_counter() - t0

        auc = m["auc"]
        mejora = not math.isnan(auc) and auc > mejor_auc
        if mejora:
            mejor_auc, sin_mejora = auc, 0
            torch.save({"modelo": resultados["modelo"], "arquitectura": resultados.get("arquitectura"),
                        "estado": modelo.state_dict(), "epoca": epoca, "auc_val": auc, "config": cfg.como_dict()},
                       carpeta / "mejor.pt")
        else:
            sin_mejora += 1

        fila = {"epoca": epoca, "perdida_tr": perdida_tr, "perdida_va": perdida_va,
                "auc": auc, "sensibilidad": m["sensibilidad"], "especificidad": m["especificidad"],
                "accuracy": m["accuracy"], **m["matriz_confusion"], "segundos": segundos,
                "mejor": mejora}
        resultados["epocas"].append(fila)
        print(f"época {epoca:>3}/{cfg.epocas_max} | pérdida tr {perdida_tr:.4f} va {perdida_va:.4f} | "
              f"AUC {auc:.3f} sens {m['sensibilidad']:.2f} espec {m['especificidad']:.2f} | "
              f"{segundos:.0f} s{'  *' if mejora else ''}")

        resultados["mejor_auc_val"] = None if mejor_auc == -math.inf else mejor_auc
        resultados["tiempo_total_s"] = time.perf_counter() - inicio
        if dispositivo.type == "cuda":
            resultados["memoria_max_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        guardar(resultados, carpeta)

        if sin_mejora >= cfg.paciencia:
            print(f"Parada: {cfg.paciencia} épocas sin mejorar el AUC por paciente.")
            break

    return resultados


def guardar(resultados: dict, carpeta: Path) -> None:
    (carpeta / "resultados.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    if resultados.get("epocas"):
        with open(carpeta / "historial.csv", "w", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=list(resultados["epocas"][0]))
            escritor.writeheader()
            escritor.writerows(resultados["epocas"])


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #

def leer_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena una CNN con evaluación por paciente.")
    p.add_argument("--modo", choices=["rapido", "completo"], required=True)
    p.add_argument("--modelo", required=True, help="nombre registrado en src/models (p. ej. cnn_v1)")
    p.add_argument("--ponderada", action="store_true", help="BCEWithLogitsLoss con pos_weight = N0/N1")
    p.add_argument("--nota", default="", help="comentario libre que se guarda con los resultados")
    # Sobrescribir valores del modo (si no se indican, se usan los de src/config.py)
    p.add_argument("--batch", dest="batch_size", type=int)
    p.add_argument("--epocas", dest="epocas_max", type=int)
    p.add_argument("--paciencia", type=int)
    p.add_argument("--lr", dest="learning_rate", type=float)
    p.add_argument("--fold", dest="fold_val", type=int)
    p.add_argument("--pacientes", dest="n_pacientes", type=int)
    p.add_argument("--workers", dest="num_workers", type=int)
    p.add_argument("--semilla", type=int)
    return p.parse_args()


def main() -> None:
    args = leer_argumentos()
    cambios = {k: v for k, v in vars(args).items() if k not in {"modo", "modelo", "ponderada", "nota"}}
    cfg = obtener_config(args.modo, **cambios)

    fijar_semilla(cfg.semilla)
    dispositivo = elegir_dispositivo()
    entrena, valida = preparar_datos(cfg)

    modelo = crear_modelo(args.modelo).to(dispositivo)
    peso_positivos = uc.pos_weight(entrena) if args.ponderada else None
    criterio = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(peso_positivos, device=dispositivo) if peso_positivos else None)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=cfg.learning_rate)

    usa_gpu = dispositivo.type == "cuda"
    opciones = {"batch_size": cfg.batch_size, "num_workers": cfg.num_workers, "pin_memory": usa_gpu,
                "persistent_workers": cfg.num_workers > 0}
    cargador_tr = DataLoader(uc.BreastDCEDataset(entrena), shuffle=True,
                             generator=torch.Generator().manual_seed(cfg.semilla), **opciones)
    cargador_va = DataLoader(uc.BreastDCEDataset(valida), shuffle=False, **opciones)

    nombre = f"{datetime.now().astimezone():%Y%m%d-%H%M%S}_{args.modelo}_{cfg.modo}{'_ponderada' if args.ponderada else ''}"
    carpeta = CARPETA_RUNS / nombre
    carpeta.mkdir(parents=True, exist_ok=True)

    resultados = {
        "ejecucion": nombre,
        "modelo": args.modelo,
        "arquitectura": VERSIONES[args.modelo],
        "parametros": contar_parametros(modelo),
        "perdida": "ponderada" if args.ponderada else "normal",
        "pos_weight": peso_positivos,
        "agregacion": METODO_AGREGACION,
        "umbral": UMBRAL,
        "nota": args.nota,
        "config": cfg.como_dict(),
        "maquina": describir_dispositivo(dispositivo),
        "datos": {
            "pacientes_entrena": int(entrena.patient_id.nunique()), "cortes_entrena": len(entrena),
            "pacientes_valida": int(valida.patient_id.nunique()), "cortes_valida": len(valida),
            "pcr_pacientes_valida": int(valida.groupby("patient_id").pCR.first().sum()),
        },
    }

    print(f"Ejecución: {nombre}")
    print(f"Modelo {args.modelo}: {resultados['parametros']:,} parámetros | dispositivo: "
          f"{resultados['maquina']['backend']} {resultados['maquina']['gpu'] or ''}")
    print(f"Entrena: {resultados['datos']['pacientes_entrena']} pacientes / {len(entrena)} cortes | "
          f"Valida: {resultados['datos']['pacientes_valida']} pacientes / {len(valida)} cortes")
    print(f"Pérdida {resultados['perdida']} | batch {cfg.batch_size} | lr {cfg.learning_rate} | "
          f"hasta {cfg.epocas_max} épocas, paciencia {cfg.paciencia}\n")

    entrenar_modelo(modelo, cargador_tr, cargador_va, valida, criterio, optimizador,
                    dispositivo, cfg, carpeta, resultados)
    print(f"\nMejor AUC por paciente en validación: {resultados['mejor_auc_val']}")
    print(f"Resultados en {carpeta.relative_to(RAIZ_REPO)}")


if __name__ == "__main__":
    main()
