"""Pruebas de sanidad: ¿el 99 % es real o hay fuga de información?

1. Etiquetas permutadas: la CNN se entrena con etiquetas al azar. Si hubiera fuga,
   seguiría acertando; sin fuga debe caer al nivel de la clase mayoritaria.
2. Línea base de umbral: solo el pico de aceleración (SMV máximo), umbral elegido
   en entrenamiento. Mide qué tan fácil es la tarea sin ninguna red.
3. Validación cruzada de 5 pliegues agrupada por sujeto: la CNN en otras particiones.

    python src/verificar_resultados.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelos  # noqa: E402
from entrenar import fijar_semilla, metricas  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "procesado"
disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def entrenar_cnn(Xtr, ytr, Xva, yva, semilla=11, epocas=20, paciencia=3):
    """Mismo protocolo que entrenar.py para la CNN de referencia."""
    fijar_semilla(semilla)
    red = modelos.crear("CNN-Casilari").to(disp)
    opt = torch.optim.SGD(red.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    f = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).long()), batch_size=64,
                    shuffle=True, generator=torch.Generator().manual_seed(semilla))
    xv, yv = torch.from_numpy(Xva).to(disp), torch.from_numpy(yva).long().to(disp)
    mejor, estado, sin = np.inf, None, 0
    for _ in range(epocas):
        red.train()
        for xb, yb in dl:
            opt.zero_grad()
            f(red(xb.to(disp)), yb.to(disp)).backward()
            opt.step()
        red.eval()
        with torch.no_grad():
            lv = f(red(xv), yv).item()
        if lv < mejor:
            mejor, sin = lv, 0
            estado = {k: v.clone() for k, v in red.state_dict().items()}
        else:
            sin += 1
            if sin >= paciencia:
                break
    red.load_state_dict(estado)
    return red


@torch.no_grad()
def probar(red, X):
    red.eval()
    return np.concatenate([torch.softmax(red(torch.from_numpy(X[i:i + 256]).to(disp)), 1)[:, 1].cpu().numpy()
                           for i in range(0, len(X), 256)])


def main():
    X0 = np.load(DATOS / "X.npy")
    meta = pd.read_csv(DATOS / "metadatos.csv")
    y = meta.etiqueta.to_numpy()
    p = meta.particion.to_numpy()
    tr, va, te = (np.where(p == k)[0] for k in ["entrenamiento", "validacion", "prueba"])
    X = X0 - X0[tr].mean(0, keepdims=True)
    out = {}

    # 1. Etiquetas permutadas (solo en entrenamiento y validación; la prueba conserva las reales)
    rng = np.random.default_rng(0)
    res = []
    for s in [11, 23, 42]:
        ytr_p, yva_p = rng.permutation(y[tr]), rng.permutation(y[va])
        red = entrenar_cnn(X[tr], ytr_p, X[va], yva_p, semilla=s)
        m = metricas(y[te], probar(red, X[te]))
        res.append({"semilla": s, "exactitud": m["exactitud"], "f1_macro": m["f1_macro"]})
    out["etiquetas_permutadas"] = res
    out["clase_mayoritaria_prueba"] = float((y[te] == 0).mean())

    # 2. Umbral sobre el SMV máximo de la traza completa
    smv = meta.smv_max_g.to_numpy()
    cand = np.unique(smv[tr])
    accs = [accuracy_score(y[tr], (smv[tr] >= u).astype(int)) for u in cand]
    u = float(cand[int(np.argmax(accs))])
    pred = (smv[te] >= u).astype(int)
    mu = metricas(y[te], pred.astype(float))
    out["umbral_smv"] = {"umbral_g": u, **{k: mu[k] for k in ["exactitud", "f1_macro", "sensibilidad",
                                                              "especificidad"]}}

    # 3. Validación cruzada de 5 pliegues agrupada por sujeto (sobre todos los datos)
    grupos = meta.sujeto.to_numpy()
    estrato = np.where(meta.sujeto.str.startswith("SE"), 2, 0) + y  # edad × clase
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    pliegues = []
    for k, (itr, ite) in enumerate(cv.split(X0, estrato, grupos)):
        sujetos_tr = np.unique(grupos[itr])
        rng2 = np.random.default_rng(k)
        s_val = rng2.choice(sujetos_tr, 2, replace=False)  # 2 sujetos de validación, como en el protocolo
        ival = itr[np.isin(grupos[itr], s_val)]
        itr2 = itr[~np.isin(grupos[itr], s_val)]
        Xk = X0 - X0[itr2].mean(0, keepdims=True)
        red = entrenar_cnn(Xk[itr2], y[itr2], Xk[ival], y[ival], semilla=11)
        m = metricas(y[ite], probar(red, Xk[ite]))
        pliegues.append({"pliegue": k + 1, "sujetos_prueba": len(np.unique(grupos[ite])),
                         "caidas_prueba": int(y[ite].sum()), "exactitud": m["exactitud"],
                         "f1_macro": m["f1_macro"], "sensibilidad": m["sensibilidad"],
                         "especificidad": m["especificidad"]})
        print(pliegues[-1], flush=True)
    out["cv_5_pliegues_por_sujeto"] = pliegues
    df = pd.DataFrame(pliegues)
    out["cv_resumen"] = {c: [float(df[c].mean()), float(df[c].std(ddof=1))]
                         for c in ["exactitud", "f1_macro", "sensibilidad", "especificidad"]}

    (RAIZ / "results" / "verificacion.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                                        encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "cv_5_pliegues_por_sujeto"}, indent=1,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
