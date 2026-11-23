"""Lanza todas las ejecuciones: cada modelo con 10 semillas distintas.

Cada ejecución es un proceso de Python nuevo con un entorno limpio, para que no
se arrastren variables, pesos ni memoria de GPU entre ejecuciones. Al final une
los CSV en results/ejecuciones.xlsx.

    python src/ejecutar_todo.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = ["CNN-Casilari", "InceptionTime"]
SEMILLAS = [11, 23, 42, 57, 89, 101, 137, 199, 256, 311]
SALIDA = RAIZ / "results" / "ejecuciones"


def entorno_limpio() -> dict:
    base = {k: os.environ[k] for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP", "USERPROFILE", "WINDIR",
                                       "COMSPEC", "PATHEXT", "LOCALAPPDATA", "APPDATA", "HOME",
                                       "USERNAME", "USERDOMAIN", "HOMEDRIVE", "HOMEPATH", "SYSTEMDRIVE",
                                       "PROGRAMDATA")
            if k in os.environ}
    base["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return base


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    for semilla in SEMILLAS:
        for modelo in MODELOS:  # se alternan para que el estado térmico de la GPU afecte igual a ambos
            if (SALIDA / f"{modelo}_s{semilla}.csv").exists():
                continue
            print(f"--> {modelo} semilla {semilla}", flush=True)
            subprocess.run([sys.executable, "-I", str(RAIZ / "src" / "entrenar.py"),
                            "--modelo", modelo, "--semilla", str(semilla)],
                           check=True, env=entorno_limpio(), cwd=RAIZ / "src")
    filas = [pd.read_csv(SALIDA / f"{m}_s{s}.csv") for m in MODELOS for s in SEMILLAS]
    tabla = pd.concat(filas, ignore_index=True)
    tabla.to_csv(RAIZ / "results" / "ejecuciones.csv", index=False)
    tabla.to_excel(RAIZ / "results" / "ejecuciones.xlsx", index=False, sheet_name="ejecuciones")
    print(tabla.groupby("modelo")[["exactitud", "f1_macro", "sensibilidad", "especificidad"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
