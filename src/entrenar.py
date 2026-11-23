"""Una ejecución: entrena un modelo con una semilla y lo evalúa en prueba.

Cada ejecución corre en un proceso nuevo (lo lanza ejecutar_todo.py) para que
no queden variables, pesos ni caché de GPU de la ejecución anterior.

    python src/entrenar.py --modelo CNN-Casilari --semilla 11
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score, confusion_matrix)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelos  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "procesado"

# Mismo presupuesto para ambos modelos (protocolo de Casilari et al., 2020)
EPOCAS_MAX = 20
PACIENCIA = 3
LOTE = 64
HIPER = {
    # Tabla 5 de Casilari et al. (SGD con momento; L2 = 1e-4 por defecto en MATLAB)
    "CNN-Casilari": {"optimizador": "SGD momento 0,9", "lr": 1e-4, "l2": 1e-4},
    # Ismail Fawaz et al. (2020): Adam, lr 1e-3
    "InceptionTime": {"optimizador": "Adam", "lr": 1e-3, "l2": 0.0},
}


def fijar_semilla(s: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cargar():
    X = np.load(DATOS / "X.npy")
    meta = pd.read_csv(DATOS / "metadatos.csv")
    y = meta["etiqueta"].to_numpy()
    idx = {p: np.where(meta["particion"] == p)[0] for p in ["entrenamiento", "validacion", "prueba"]}
    # 'zerocenter' (como imageInputLayer de MATLAB): resta la media del entrenamiento
    media = X[idx["entrenamiento"]].mean(axis=0, keepdims=True)
    X = X - media
    return X, y, meta, idx


def metricas(y, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "exactitud": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "f1_caida": f1_score(y, pred),
        "sensibilidad": recall_score(y, pred),
        "especificidad": tn / (tn + fp),
        "precision": precision_score(y, pred, zero_division=0),
        "exactitud_balanceada": balanced_accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, prob),
        "VP": int(tp), "FN": int(fn), "FP": int(fp), "VN": int(tn),
    }


@torch.no_grad()
def predecir(modelo, X, disp, lote=256):
    modelo.eval()
    salida = []
    for i in range(0, len(X), lote):
        xb = torch.from_numpy(X[i:i + lote]).to(disp)
        salida.append(torch.softmax(modelo(xb), 1)[:, 1].cpu().numpy())
    return np.concatenate(salida)


@torch.no_grad()
def latencia_ms(modelo, x1, disp, repeticiones=300):
    modelo.eval()
    x1 = x1.to(disp)
    for _ in range(30):
        modelo(x1)
    if disp.type == "cuda":
        torch.cuda.synchronize()
    tiempos = []
    for _ in range(repeticiones):
        t = time.perf_counter()
        modelo(x1)
        if disp.type == "cuda":
            torch.cuda.synchronize()
        tiempos.append((time.perf_counter() - t) * 1000)
    return float(np.median(tiempos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True, choices=list(HIPER))
    ap.add_argument("--semilla", type=int, required=True)
    ap.add_argument("--salida", default=str(RAIZ / "results" / "ejecuciones"))
    a = ap.parse_args()
    fijar_semilla(a.semilla)
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.modelo}_s{a.semilla}"

    X, y, meta, idx = cargar()
    tr, va, te = idx["entrenamiento"], idx["validacion"], idx["prueba"]
    g = torch.Generator().manual_seed(a.semilla)
    dl = DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr]).long()),
                    batch_size=LOTE, shuffle=True, generator=g)

    modelo = modelos.crear(a.modelo).to(disp)
    h = HIPER[a.modelo]
    if a.modelo == "CNN-Casilari":
        opt = torch.optim.SGD(modelo.parameters(), lr=h["lr"], momentum=0.9, weight_decay=h["l2"])
    else:
        opt = torch.optim.Adam(modelo.parameters(), lr=h["lr"], weight_decay=h["l2"])
    perdida = nn.CrossEntropyLoss()

    hist, mejor, mejor_ep, sin_mejora, estado = [], np.inf, 0, 0, None
    Xva = torch.from_numpy(X[va]).to(disp)
    yva = torch.from_numpy(y[va]).long().to(disp)
    if disp.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(1, EPOCAS_MAX + 1):
        modelo.train()
        suma, n, aciertos = 0.0, 0, 0
        for xb, yb in dl:
            xb, yb = xb.to(disp), yb.to(disp)
            opt.zero_grad()
            o = modelo(xb)
            l = perdida(o, yb)
            l.backward()
            opt.step()
            suma += l.item() * len(yb)
            n += len(yb)
            aciertos += (o.argmax(1) == yb).sum().item()
        modelo.eval()
        with torch.no_grad():
            ov = modelo(Xva)
            lv = perdida(ov, yva).item()
            av = (ov.argmax(1) == yva).float().mean().item()
        hist.append({"epoca": ep, "perdida_entr": suma / n, "exactitud_entr": aciertos / n,
                     "perdida_val": lv, "exactitud_val": av})
        if lv < mejor - 1e-6:
            mejor, mejor_ep, sin_mejora = lv, ep, 0
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break
    if disp.type == "cuda":
        torch.cuda.synchronize()
    t_entr = time.perf_counter() - t0
    modelo.load_state_dict(estado)

    prob = predecir(modelo, X[te], disp)
    m = metricas(y[te], prob)
    x1 = torch.from_numpy(X[te[:1]])
    lat_gpu = latencia_ms(modelo, x1, disp) if disp.type == "cuda" else float("nan")
    modelo_cpu = modelos.crear(a.modelo)
    modelo_cpu.load_state_dict({k: v.cpu() for k, v in estado.items()})
    torch.set_num_threads(1)
    lat_cpu = latencia_ms(modelo_cpu, x1, torch.device("cpu"), repeticiones=100)
    params = sum(p.numel() for p in modelo.parameters())
    macs = modelos.contar_macs(modelo_cpu.eval(), x1)

    fila = {"modelo": a.modelo, "semilla": a.semilla, "epoca_final": len(hist), "mejor_epoca": mejor_ep,
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()},
            "tiempo_entrenamiento_s": round(t_entr, 2), "latencia_gpu_ms": round(lat_gpu, 4),
            "latencia_cpu_ms": round(lat_cpu, 4), "parametros": params, "macs": macs,
            "tamano_mb": round(params * 4 / 2 ** 20, 3), "dispositivo": torch.cuda.get_device_name(0)
            if disp.type == "cuda" else "cpu", "torch": torch.__version__}
    pd.DataFrame([fila]).to_csv(out / f"{tag}.csv", index=False)
    pd.DataFrame(hist).to_csv(out / f"{tag}_historia.csv", index=False)
    pd.DataFrame({"archivo": meta.loc[te, "archivo"].to_numpy(), "y": y[te], "prob_caida": prob}) \
        .to_csv(out / f"{tag}_predicciones.csv", index=False)
    torch.save(estado, out / f"{tag}.pt")
    print(json.dumps(fila, ensure_ascii=False))


if __name__ == "__main__":
    main()
