"""Prepara SisFall: ventanas de ±2,5 s alrededor del pico de aceleración y
partición 70/25/5 (entrenamiento/prueba/validación) agrupada por sujeto.

Uso:
    python data/preparar_datos.py --raw data/raw/SisFall --out data/procesado

Siguiendo a Casilari et al. (2020), la entrada de la red es la señal triaxial
del acelerómetro ADXL345 (columnas 1-3 de SisFall) en g, recortada a una
ventana de ±2,5 s (1001 muestras a 200 Hz) centrada en el máximo de la
magnitud SMV de la traza.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

FS = 200                      # Hz, frecuencia de muestreo de SisFall
MEDIA_VENTANA = int(2.5 * FS)  # 500 muestras a cada lado del pico -> 1001
ESCALA_ADXL345 = (2 * 16) / (2 ** 13)  # g por cuenta (±16 g, 13 bits), README de SisFall
PATRON = re.compile(r"^(?P<act>[DF]\d{2})_(?P<suj>S[AE]\d{2})_(?P<rep>R\d{2})\.txt$")
SEMILLA_PARTICION = 2026


def leer_traza(ruta: Path) -> np.ndarray:
    """Devuelve la aceleración triaxial (n, 3) en g del ADXL345."""
    filas = []
    for linea in ruta.read_text(encoding="latin-1").splitlines():
        linea = linea.strip().rstrip(";").strip()
        if not linea:
            continue
        valores = [v.strip() for v in linea.split(",")]
        if len(valores) < 3:
            continue
        filas.append([float(v) for v in valores[:3]])
    return np.asarray(filas, dtype=np.float32) * ESCALA_ADXL345


def ventana_pico(acc: np.ndarray) -> tuple[np.ndarray, int]:
    smv = np.sqrt((acc ** 2).sum(axis=1))
    t0 = int(np.argmax(smv))
    rellena = np.pad(acc, ((MEDIA_VENTANA, MEDIA_VENTANA), (0, 0)), mode="edge")
    v = rellena[t0: t0 + 2 * MEDIA_VENTANA + 1]
    return v.T.copy(), t0  # (3, 1001)


def particion_por_sujeto(sujetos: pd.DataFrame) -> dict[str, str]:
    """70/25/5 por sujeto, estratificada por grupo de edad (SA jóvenes / SE mayores).

    Los jóvenes (SA) hicieron caídas y actividades; los mayores (SE), salvo SE06,
    solo actividades. Estratificar por grupo mantiene la proporción de caídas.
    """
    rng = np.random.default_rng(SEMILLA_PARTICION)
    asignacion = {}
    for grupo, sub in sujetos.groupby("grupo"):
        ids = sorted(sub["sujeto"].tolist())
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, round(0.05 * n))
        n_test = round(0.25 * n)
        for i, s in enumerate(ids):
            asignacion[s] = "validacion" if i < n_val else ("prueba" if i < n_val + n_test else "entrenamiento")
    return asignacion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/SisFall")
    ap.add_argument("--out", default="data/procesado")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    registros, ventanas = [], []
    for ruta in sorted(raw.glob("S[AE]*/*.txt")):
        m = PATRON.match(ruta.name)
        if not m:
            continue
        acc = leer_traza(ruta)
        if len(acc) < FS:  # trazas vacías o corruptas
            print("omitida (muy corta):", ruta.name)
            continue
        v, t0 = ventana_pico(acc)
        registros.append({
            "archivo": ruta.name,
            "actividad": m["act"],
            "sujeto": m["suj"],
            "repeticion": m["rep"],
            "grupo": "SE (mayores)" if m["suj"].startswith("SE") else "SA (jóvenes)",
            "etiqueta": 1 if m["act"].startswith("F") else 0,
            "muestras": len(acc),
            "indice_pico": t0,
            "smv_max_g": float(np.sqrt((acc ** 2).sum(1)).max()),
            "hash": hashlib.sha1(v.tobytes()).hexdigest(),
        })
        ventanas.append(v)

    meta = pd.DataFrame(registros)
    X = np.stack(ventanas).astype(np.float32)
    sujetos = meta[["sujeto", "grupo"]].drop_duplicates()
    asign = particion_por_sujeto(sujetos)
    meta["particion"] = meta["sujeto"].map(asign)

    # Control de fugas: ningún sujeto ni ventana idéntica en dos particiones
    por_sujeto = meta.groupby("sujeto")["particion"].nunique()
    assert (por_sujeto == 1).all(), "Un sujeto quedó en dos particiones"
    dup = meta.groupby("hash")["particion"].nunique()
    n_dup_cruzados = int((dup > 1).sum())
    n_dup_total = int(meta["hash"].duplicated().sum())

    np.save(out / "X.npy", X)
    meta.to_csv(out / "metadatos.csv", index=False)

    resumen = (meta.groupby("particion")
               .agg(trazas=("archivo", "size"), sujetos=("sujeto", "nunique"),
                    caidas=("etiqueta", "sum"))
               .assign(actividades=lambda d: d["trazas"] - d["caidas"]))
    resumen["porc_trazas"] = (100 * resumen["trazas"] / resumen["trazas"].sum()).round(1)
    resumen["porc_sujetos"] = (100 * resumen["sujetos"] / resumen["sujetos"].sum()).round(1)
    resumen.to_csv(out / "resumen_particion.csv")
    info = {
        "trazas": len(meta), "sujetos": int(meta["sujeto"].nunique()),
        "caidas": int(meta["etiqueta"].sum()), "actividades": int((meta["etiqueta"] == 0).sum()),
        "forma_X": list(X.shape), "duplicados_exactos": n_dup_total,
        "duplicados_entre_particiones": n_dup_cruzados,
        "semilla_particion": SEMILLA_PARTICION,
        "sujetos_por_particion": {p: sorted(g["sujeto"].unique().tolist())
                                  for p, g in meta.groupby("particion")},
    }
    (out / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(resumen.to_string())
    print(json.dumps({k: v for k, v in info.items() if k != "sujetos_por_particion"}, indent=2))


if __name__ == "__main__":
    main()
