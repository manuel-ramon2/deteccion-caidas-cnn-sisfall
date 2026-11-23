"""Análisis de las ejecuciones: tabla media ± DE, pruebas estadísticas, curvas,
matrices de confusión, desempeño contra latencia, análisis de errores y Grad-CAM.

    python src/analizar.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelos  # noqa: E402
from entrenar import cargar  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "results"
EJ = RES / "ejecuciones"
FIG = RES / "figuras"
FIG.mkdir(parents=True, exist_ok=True)
MODELOS = ["CNN-Casilari", "InceptionTime"]
COLOR = {"CNN-Casilari": "#2a6fdb", "InceptionTime": "#e07b39"}
plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "savefig.bbox": "tight"})

ACTIVIDADES = {
    "D01": "Caminar despacio", "D02": "Caminar rápido", "D03": "Trotar despacio", "D04": "Trotar rápido",
    "D05": "Escaleras despacio", "D06": "Escaleras rápido", "D07": "Sentarse silla media (lento)",
    "D08": "Sentarse silla media (rápido)", "D09": "Sentarse silla baja (lento)",
    "D10": "Sentarse silla baja (rápido)", "D11": "Colapsar en silla al levantarse",
    "D12": "Acostarse despacio", "D13": "Acostarse rápido", "D14": "Girar acostado",
    "D15": "Agacharse doblando rodillas", "D16": "Agacharse sin doblar rodillas",
    "D17": "Entrar y salir de un carro", "D18": "Tropezar caminando", "D19": "Saltar suave",
    "F01": "Adelante, resbalón caminando", "F02": "Atrás, resbalón caminando", "F03": "Lateral, resbalón caminando",
    "F04": "Adelante, tropiezo caminando", "F05": "Adelante, tropiezo trotando", "F06": "Vertical, desmayo",
    "F07": "Desmayo apoyando manos", "F08": "Adelante al levantarse", "F09": "Lateral al levantarse",
    "F10": "Adelante al sentarse", "F11": "Atrás al sentarse", "F12": "Lateral al sentarse",
    "F13": "Adelante sentado (desmayo)", "F14": "Atrás sentado (desmayo)", "F15": "Lateral sentado (desmayo)",
}
METRICAS = ["exactitud", "f1_macro", "sensibilidad", "especificidad", "precision", "roc_auc"]


def resumen(t: pd.DataFrame) -> pd.DataFrame:
    cols = METRICAS + ["epoca_final", "tiempo_entrenamiento_s", "latencia_gpu_ms", "latencia_cpu_ms"]
    g = t.groupby("modelo")[cols]
    out = pd.DataFrame(index=MODELOS)
    for c in cols:
        m, s = g.mean()[c], g.std(ddof=1)[c]
        out[c] = [f"{m[k]:.4f} ± {s[k]:.4f}" for k in MODELOS]
    out["parametros"] = t.groupby("modelo")["parametros"].first()
    out["macs"] = t.groupby("modelo")["macs"].first()
    out["tamano_mb"] = t.groupby("modelo")["tamano_mb"].first()
    return out


def pruebas(t: pd.DataFrame) -> dict:
    a = t[t.modelo == MODELOS[0]].sort_values("semilla")
    b = t[t.modelo == MODELOS[1]].sort_values("semilla")
    out = {}
    for c in METRICAS + ["latencia_gpu_ms", "latencia_cpu_ms", "tiempo_entrenamiento_s"]:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        d = y - x
        try:
            w = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            wp, ws = float(w.pvalue), float(w.statistic)
        except ValueError:
            wp, ws = 1.0, float("nan")
        tt = stats.ttest_rel(x, y)
        sw = stats.shapiro(d) if np.ptp(d) > 0 else None
        out[c] = {"media_cnn": x.mean(), "media_inception": y.mean(), "dif_media": d.mean(),
                  "wilcoxon_W": ws, "wilcoxon_p": wp, "t_pareada_t": float(tt.statistic),
                  "t_pareada_p": float(tt.pvalue), "shapiro_p_dif": float(sw.pvalue) if sw else None,
                  "d_cohen_pareada": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None}
    return out


def mcnemar_por_semilla(t):
    filas = []
    for s in sorted(t.semilla.unique()):
        pa = pd.read_csv(EJ / f"{MODELOS[0]}_s{s}_predicciones.csv")
        pb = pd.read_csv(EJ / f"{MODELOS[1]}_s{s}_predicciones.csv")
        ca = (pa.prob_caida >= .5).astype(int) == pa.y
        cb = (pb.prob_caida >= .5).astype(int) == pb.y
        tabla = [[int((ca & cb).sum()), int((ca & ~cb).sum())], [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
        r = mcnemar(tabla, exact=True)
        filas.append({"semilla": s, "ambos_aciertan": tabla[0][0], "solo_cnn_acierta": tabla[0][1],
                      "solo_inception_acierta": tabla[1][0], "ambos_fallan": tabla[1][1], "p_mcnemar": r.pvalue})
    return pd.DataFrame(filas)


def fig_curvas(t):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for m in MODELOS:
        hs = [pd.read_csv(EJ / f"{m}_s{s}_historia.csv") for s in sorted(t.semilla.unique())]
        L = max(len(h) for h in hs)
        for j, (col, nom) in enumerate([("perdida", "Pérdida"), ("exactitud", "Exactitud")]):
            for parte, est in [("entr", "-"), ("val", "--")]:
                M = np.full((len(hs), L), np.nan)
                for i, h in enumerate(hs):
                    M[i, :len(h)] = h[f"{col}_{parte}"]
                mu, sd = np.nanmean(M, 0), np.nanstd(M, 0)
                ep = np.arange(1, L + 1)
                ax[j].plot(ep, mu, est, color=COLOR[m], lw=1.2,
                           label=f"{m} ({'entren.' if parte == 'entr' else 'valid.'})")
                ax[j].fill_between(ep, mu - sd, mu + sd, color=COLOR[m], alpha=.12)
            ax[j].set_xlabel("Época")
            ax[j].set_ylabel(nom)
            ax[j].grid(alpha=.3)
    ax[1].legend(fontsize=6.5, loc="lower right")
    fig.savefig(FIG / "fig_curvas_aprendizaje.png")
    plt.close(fig)


def fig_confusion(t):
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.5))
    for k, m in enumerate(MODELOS):
        sub = t[t.modelo == m]
        C = np.array([[sub.VN.sum(), sub.FP.sum()], [sub.FN.sum(), sub.VP.sum()]]) / len(sub)
        Cn = C / C.sum(1, keepdims=True)
        ax[k].imshow(Cn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax[k].text(j, i, f"{C[i, j]:.1f}\n({100 * Cn[i, j]:.2f} %)", ha="center", va="center",
                           color="white" if Cn[i, j] > .5 else "black", fontsize=8)
        ax[k].set_xticks([0, 1], ["ADL", "Caída"])
        ax[k].set_yticks([0, 1], ["ADL", "Caída"])
        ax[k].set_xlabel("Predicción")
        ax[k].set_ylabel("Real")
        ax[k].set_title(m, fontsize=9)
    fig.savefig(FIG / "fig_matrices_confusion.png")
    plt.close(fig)


def fig_latencia(t):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for j, (col, nom) in enumerate([("latencia_gpu_ms", "Latencia GPU (ms, lote 1)"),
                                    ("latencia_cpu_ms", "Latencia CPU 1 hilo (ms, lote 1)")]):
        for m in MODELOS:
            s = t[t.modelo == m]
            ax[j].scatter(s[col], s.f1_macro, color=COLOR[m], s=14, alpha=.6, label=m)
            ax[j].errorbar(s[col].mean(), s.f1_macro.mean(), xerr=s[col].std(), yerr=s.f1_macro.std(),
                           color=COLOR[m], marker="D", ms=6, mec="black", capsize=3)
        ax[j].set_xlabel(nom)
        ax[j].set_ylabel("F1 macro (prueba)")
        ax[j].grid(alpha=.3)
    ax[0].legend(fontsize=7)
    fig.savefig(FIG / "fig_desempeno_vs_latencia.png")
    plt.close(fig)


def fig_cajas(t):
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.3))
    for j, c in enumerate(["exactitud", "f1_macro", "sensibilidad", "especificidad"]):
        data = [t[t.modelo == m][c] for m in MODELOS]
        bp = ax[j].boxplot(data, patch_artist=True, widths=.5)
        for p, m in zip(bp["boxes"], MODELOS):
            p.set_facecolor(COLOR[m])
            p.set_alpha(.5)
        for i, d in enumerate(data):
            ax[j].scatter(np.full(len(d), i + 1) + np.random.default_rng(0).uniform(-.08, .08, len(d)),
                          d, s=8, color="black", zorder=3)
        ax[j].set_xticks([1, 2], ["CNN", "Incep."])
        ax[j].set_title(c.replace("_", " ").capitalize(), fontsize=8)
        ax[j].grid(alpha=.3)
    fig.savefig(FIG / "fig_distribucion_metricas.png")
    plt.close(fig)


def errores(t, meta):
    """Tasa de error por actividad y por grupo de edad, promediando las semillas."""
    filas = []
    for m in MODELOS:
        for s in sorted(t.semilla.unique()):
            p = pd.read_csv(EJ / f"{m}_s{s}_predicciones.csv")
            p["error"] = ((p.prob_caida >= .5).astype(int) != p.y).astype(int)
            p["modelo"], p["semilla"] = m, s
            filas.append(p)
    P = pd.concat(filas).merge(meta[["archivo", "actividad", "sujeto", "grupo", "smv_max_g"]], on="archivo")
    por_act = (P.groupby(["modelo", "actividad"])["error"].mean().unstack(0) * 100).round(2)
    por_act["descripcion"] = por_act.index.map(ACTIVIDADES)
    por_grupo = (P.groupby(["modelo", "grupo", "y"])["error"].mean().unstack(0) * 100).round(2)
    por_sujeto = (P.groupby(["modelo", "sujeto"])["error"].mean().unstack(0) * 100).round(2)
    por_act.to_csv(RES / "errores_por_actividad.csv")
    por_grupo.to_csv(RES / "errores_por_grupo.csv")
    por_sujeto.to_csv(RES / "errores_por_sujeto.csv")

    top = por_act[MODELOS].max(axis=1).sort_values(ascending=False).head(10).index
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    x = np.arange(len(top))
    for k, m in enumerate(MODELOS):
        ax.bar(x + (k - .5) * .38, por_act.loc[top, m], .38, color=COLOR[m], label=m)
    ax.set_xticks(x, [f"{a}\n{ACTIVIDADES[a][:22]}" for a in top], fontsize=6, rotation=0)
    ax.set_ylabel("Tasa de error (%)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=.3)
    fig.savefig(FIG / "fig_errores_por_actividad.png")
    plt.close(fig)

    # Error por sujeto de prueba
    suj = por_sujeto.copy()
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    x = np.arange(len(suj))
    for k, m in enumerate(MODELOS):
        ax.bar(x + (k - .5) * .38, suj[m], .38, color=COLOR[m], label=m)
    ax.set_xticks(x, suj.index, fontsize=7)
    ax.set_ylabel("Tasa de error (%)")
    ax.set_xlabel("Sujeto de prueba (SA: 19-30 años; SE: 60-75 años)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=.3)
    fig.savefig(FIG / "fig_errores_por_sujeto.png")
    plt.close(fig)

    # SMV máximo de aciertos y errores (¿se confunden los impactos fuertes?)
    smv = P.groupby(["modelo", "y", "error"])["smv_max_g"].median().unstack([1, 2]).round(2)
    smv.to_csv(RES / "smv_aciertos_errores.csv")
    return P, por_act, por_grupo


def gradcam(t, meta, P):
    """Grad-CAM 1D sobre la última capa convolucional de la CNN de referencia."""
    X, y, _, idx = cargar()
    s_med = int(t[t.modelo == MODELOS[0]].sort_values("f1_macro").iloc[len(t[t.modelo == MODELOS[0]]) // 2].semilla)
    red = modelos.crear(MODELOS[0])
    red.load_state_dict(torch.load(EJ / f"{MODELOS[0]}_s{s_med}.pt", map_location="cpu"))
    red.eval()
    for capa_relu in red.modules():  # los ganchos de gradiente no admiten ReLU en el sitio
        if isinstance(capa_relu, torch.nn.ReLU):
            capa_relu.inplace = False
    act, grad = {}, {}
    capa = red.capa_gradcam
    capa.register_forward_hook(lambda m, i, o: act.__setitem__("a", o))
    capa.register_full_backward_hook(lambda m, gi, go: grad.__setitem__("g", go[0]))

    Xraw = np.load(RAIZ / "data" / "procesado" / "X.npy")
    pred = pd.read_csv(EJ / f"{MODELOS[0]}_s{s_med}_predicciones.csv")
    pred = pred.merge(meta[["archivo", "actividad"]], on="archivo")
    pred["pred"] = (pred.prob_caida >= .5).astype(int)
    casos = []
    for nombre, filtro in [("Caída detectada (VP)", (pred.y == 1) & (pred.pred == 1)),
                           ("ADL correcta (VN)", (pred.y == 0) & (pred.pred == 0)),
                           ("Caída no detectada (FN)", (pred.y == 1) & (pred.pred == 0)),
                           ("Falsa alarma (FP)", (pred.y == 0) & (pred.pred == 1))]:
        sub = pred[filtro]
        if len(sub):
            fila = sub.iloc[np.argsort(np.abs(sub.prob_caida.to_numpy() - (1 if 'V' in nombre else .5)))[0]] \
                if "V" in nombre else sub.iloc[0]
            casos.append((nombre, fila))
    fig, axs = plt.subplots(len(casos), 1, figsize=(7.2, 1.55 * len(casos)), sharex=True)
    axs = np.atleast_1d(axs)
    tiempo = (np.arange(1001) - 500) / 200
    for ax, (nombre, fila) in zip(axs, casos):
        i = int(meta.index[meta.archivo == fila.archivo][0])
        x = torch.from_numpy(X[i:i + 1]).requires_grad_(True)
        out = red(x)
        red.zero_grad()
        out[0, 1].backward()
        w = grad["g"].mean(-1, keepdim=True)
        cam = torch.relu((w * act["a"]).sum(1))[0].detach().numpy()
        cam = np.interp(np.linspace(0, len(cam) - 1, 1001), np.arange(len(cam)), cam)
        cam = cam / (cam.max() + 1e-9)
        ax.imshow(cam[None, :], aspect="auto", cmap="jet", alpha=.35,
                  extent=[tiempo[0], tiempo[-1], -4, 4])
        for c, n in zip(range(3), ["Ax", "Ay", "Az"]):
            ax.plot(tiempo, Xraw[i, c], lw=.7, label=n)
        ax.set_ylim(-4, 4)
        ax.set_ylabel("g")
        ax.set_title(f"{nombre}: {fila.archivo[:-4]} ({ACTIVIDADES[fila.actividad]}), "
                     f"P(caída)={fila.prob_caida:.3f}", fontsize=7.5)
    axs[0].legend(fontsize=6, loc="upper right", ncol=3)
    axs[-1].set_xlabel("Tiempo relativo al pico de aceleración (s)")
    fig.savefig(FIG / "fig_gradcam_cnn.png")
    plt.close(fig)
    return s_med


def fig_arquitecturas():
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 3.6))
    def cajas(ax, bloques, titulo):
        ax.set_xlim(0, len(bloques))
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(titulo, fontsize=8.5, loc="left")
        for k, (txt, col) in enumerate(bloques):
            ax.add_patch(plt.Rectangle((k + .06, .18), .88, .64, fc=col, ec="black", lw=.6))
            ax.text(k + .5, .5, txt, ha="center", va="center", fontsize=5.6)
            if k < len(bloques) - 1:
                ax.annotate("", (k + 1.06, .5), (k + .94, .5), arrowprops=dict(arrowstyle="->", lw=.6))
    cajas(axs[0], [("Entrada\n3×1001\n(Ax,Ay,Az\n±2,5 s)", "#eeeeee"),
                   ("Conv 1×5, 16\nBN+ReLU\nMaxPool 5", "#cfe0fb"), ("Conv 1×5, 32\nBN+ReLU\nMaxPool 5", "#cfe0fb"),
                   ("Conv 1×5, 64\nBN+ReLU\nMaxPool 5", "#cfe0fb"), ("Conv 1×5, 128\nBN+ReLU", "#cfe0fb"),
                   ("Aplanar\n128×8\nDropout 0,5", "#fde7c8"), ("Densa 2\nSoftmax", "#d8f0d2"),
                   ("ADL /\nCaída", "#eeeeee")],
          "(a) CNN de referencia (Casilari et al., 2020)")
    cajas(axs[1], [("Entrada\n3×1001", "#eeeeee"),
                   ("Inception ×3\ncuello 1×1 (32)\nconv 39/19/9\n+maxpool·1×1", "#fbd9c4"),
                   ("Suma\nresidual\n+ReLU", "#f3f3c9"),
                   ("Inception ×3\n128 canales", "#fbd9c4"), ("Suma\nresidual\n+ReLU", "#f3f3c9"),
                   ("Promedio\nglobal\n(GAP)", "#fde7c8"), ("Densa 2\nSoftmax", "#d8f0d2"), ("ADL /\nCaída", "#eeeeee")],
          "(b) InceptionTime (Ismail Fawaz et al., 2020)")
    fig.savefig(FIG / "fig_arquitecturas.png")
    plt.close(fig)


def fig_datos(meta):
    X = np.load(RAIZ / "data" / "procesado" / "X.npy")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3), sharey=True)
    tiempo = (np.arange(1001) - 500) / 200
    for a, arch in zip(ax, ["D06_SA01_R01.txt", "F04_SA01_R01.txt"]):
        i = int(meta.index[meta.archivo == arch][0])
        for c, n in enumerate(["Ax", "Ay", "Az"]):
            a.plot(tiempo, X[i, c], lw=.7, label=n)
        a.plot(tiempo, np.sqrt((X[i] ** 2).sum(0)), "k", lw=.8, label="SMV")
        a.set_title(f"{arch[:-4]}: {ACTIVIDADES[arch[:3]]}", fontsize=8)
        a.set_xlabel("Tiempo relativo al pico (s)")
        a.grid(alpha=.3)
    ax[0].set_ylabel("Aceleración (g)")
    ax[1].legend(fontsize=6, ncol=4)
    fig.savefig(FIG / "fig_ejemplos_senal.png")
    plt.close(fig)


def main():
    t = pd.read_csv(RES / "ejecuciones.csv")
    _, _, meta, _ = cargar()
    r = resumen(t)
    r.to_csv(RES / "resumen_media_de.csv")
    pr = pruebas(t)
    (RES / "pruebas_estadisticas.json").write_text(json.dumps(pr, indent=2, ensure_ascii=False), encoding="utf-8")
    mc = mcnemar_por_semilla(t)
    mc.to_csv(RES / "mcnemar_por_semilla.csv", index=False)
    fig_curvas(t)
    fig_confusion(t)
    fig_latencia(t)
    fig_cajas(t)
    fig_arquitecturas()
    fig_datos(meta)
    P, por_act, por_grupo = errores(t, meta)
    s_med = gradcam(t, meta, P)
    with pd.ExcelWriter(RES / "ejecuciones.xlsx") as xw:
        t.to_excel(xw, sheet_name="ejecuciones", index=False)
        r.to_excel(xw, sheet_name="resumen_media_de")
        pd.DataFrame(pr).T.to_excel(xw, sheet_name="pruebas_estadisticas")
        mc.to_excel(xw, sheet_name="mcnemar", index=False)
        por_act.to_excel(xw, sheet_name="errores_actividad")
        por_grupo.to_excel(xw, sheet_name="errores_grupo")
    print(r.T.to_string())
    print(json.dumps({k: {kk: round(vv, 5) if isinstance(vv, float) else vv for kk, vv in v.items()}
                      for k, v in pr.items()}, indent=1, ensure_ascii=False))
    print(mc.to_string())
    print(por_grupo.to_string())
    print(por_act.sort_values(MODELOS[0], ascending=False).head(12).to_string())
    print("semilla Grad-CAM:", s_med)


if __name__ == "__main__":
    main()
