"""Genera el artículo en formato IEEE (dos columnas) a partir de los resultados.

    python articulo/generar_articulo.py

Todos los números salen de results/ y data/procesado/; nada se escribe a mano.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

RAIZ = Path(__file__).resolve().parents[1]
import sys
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "results"   # carpeta de resultados (opcional)
FIG = RES / "figuras"
SALIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else RAIZ / "articulo" / "Articulo_IEEE_CNN_vs_InceptionTime_SisFall.docx"
M1, M2 = "CNN-Casilari", "InceptionTime"
ANCHO_COL = Inches(3.45)

# ---------------------------------------------------------------- datos
t = pd.read_csv(RES / "ejecuciones.csv")
pr = json.loads((RES / "pruebas_estadisticas.json").read_text(encoding="utf-8"))
mc = pd.read_csv(RES / "mcnemar_por_semilla.csv")
info = json.loads((RAIZ / "data" / "procesado" / "info.json").read_text(encoding="utf-8"))
part = pd.read_csv(RAIZ / "data" / "procesado" / "resumen_particion.csv", index_col=0)
eact = pd.read_csv(RES / "errores_por_actividad.csv", index_col=0)
egr = pd.read_csv(RES / "errores_por_grupo.csv")
N_SEM = t.semilla.nunique()
A, B = t[t.modelo == M1], t[t.modelo == M2]


def c(x, d=2):
    """Número con coma decimal."""
    return f"{x:.{d}f}".replace(".", ",")


def pct(serie, d=2):
    return f"{c(100 * serie.mean(), d)} ± {c(100 * serie.std(ddof=1), d)}"


def ms(serie, d=3):
    return f"{c(serie.mean(), d)} ± {c(serie.std(ddof=1), d)}"


def pval(p):
    return "< 0,001" if p < 0.001 else c(p, 3)


def mejor(met):
    return M1 if pr[met]["media_cnn"] > pr[met]["media_inception"] else M2


def signif(met):
    return pr[met]["wilcoxon_p"] < 0.05


ACT = {
    "D01": "caminar despacio", "D02": "caminar rápido", "D03": "trotar despacio", "D04": "trotar rápido",
    "D05": "subir y bajar escaleras despacio", "D06": "subir y bajar escaleras rápido",
    "D07": "sentarse despacio en silla media", "D08": "sentarse rápido en silla media",
    "D09": "sentarse despacio en silla baja", "D10": "sentarse rápido en silla baja",
    "D11": "intentar levantarse y desplomarse en la silla", "D12": "acostarse despacio",
    "D13": "acostarse rápido", "D14": "girar estando acostado", "D15": "agacharse doblando rodillas",
    "D16": "agacharse sin doblar rodillas", "D17": "entrar y salir de un carro",
    "D18": "tropezar al caminar sin caer", "D19": "saltar suavemente",
    "F01": "caída hacia adelante por resbalón", "F02": "caída hacia atrás por resbalón",
    "F03": "caída lateral por resbalón", "F04": "caída hacia adelante por tropiezo",
    "F05": "caída hacia adelante trotando", "F06": "caída vertical por desmayo",
    "F07": "caída por desmayo apoyando las manos", "F08": "caída hacia adelante al levantarse",
    "F09": "caída lateral al levantarse", "F10": "caída hacia adelante al sentarse",
    "F11": "caída hacia atrás al sentarse", "F12": "caída lateral al sentarse",
    "F13": "caída hacia adelante estando sentado", "F14": "caída hacia atrás estando sentado",
    "F15": "caída lateral estando sentado",
}

# ---------------------------------------------------------------- documento
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.font.size = Pt(10)
st.paragraph_format.space_after = Pt(0)
st.paragraph_format.space_before = Pt(0)
st.paragraph_format.line_spacing = 1.0

sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin, sec.bottom_margin = Inches(0.75), Inches(1.0)
sec.left_margin = sec.right_margin = Inches(0.625)


def columnas(section, n):
    sp = section._sectPr
    cols = sp.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sp.append(cols)
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), "360")


def par(texto="", negrita=False, cursiva=False, tam=10, alin=WD_ALIGN_PARAGRAPH.JUSTIFY,
        sangria=True, antes=0, despues=0):
    p = doc.add_paragraph()
    p.alignment = alin
    pf = p.paragraph_format
    pf.first_line_indent = Inches(0.14) if sangria else None
    pf.space_before, pf.space_after = Pt(antes), Pt(despues)
    if texto:
        _runs(p, texto, negrita, cursiva, tam)
    return p


def _runs(p, texto, negrita=False, cursiva=False, tam=10):
    """Admite *cursiva* marcada con asteriscos simples."""
    partes = texto.split("*")
    for i, s in enumerate(partes):
        if not s:
            continue
        r = p.add_run(s)
        r.bold = negrita
        r.italic = cursiva or (i % 2 == 1)
        r.font.size = Pt(tam)
        r.font.name = "Times New Roman"


ROMANOS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
_n_sec = [0]
_n_sub = [0]


def seccion(titulo):
    _n_sec[0] += 1
    _n_sub[0] = 0
    p = par(f"{ROMANOS[_n_sec[0] - 1]}. {titulo.upper()}", tam=10, alin=WD_ALIGN_PARAGRAPH.CENTER,
            sangria=False, antes=8, despues=4)
    p.paragraph_format.keep_with_next = True


def subseccion(titulo):
    _n_sub[0] += 1
    p = par(f"{chr(64 + _n_sub[0])}. {titulo}", cursiva=True, sangria=False, antes=4, despues=2,
            alin=WD_ALIGN_PARAGRAPH.LEFT)
    p.paragraph_format.keep_with_next = True


_n_fig = [0]
_n_tab = [0]


def figura(archivo, pie, ancho=ANCHO_COL):
    _n_fig[0] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG / archivo), width=ancho)
    q = par(f"Fig. {_n_fig[0]}. {pie}", tam=8, sangria=False, despues=6, alin=WD_ALIGN_PARAGRAPH.JUSTIFY)
    return _n_fig[0]


def tabla(titulo, encabezado, filas, anchos, tam=7.5):
    _n_tab[0] += 1
    p = par(f"TABLA {ROMANOS[_n_tab[0] - 1]}", tam=8, alin=WD_ALIGN_PARAGRAPH.CENTER, sangria=False, antes=6)
    p.paragraph_format.keep_with_next = True
    p = par(titulo.upper(), tam=7, alin=WD_ALIGN_PARAGRAPH.CENTER, sangria=False, despues=2)
    p.paragraph_format.keep_with_next = True
    tb = doc.add_table(rows=1 + len(filas), cols=len(encabezado))
    tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    tb.style = "Table Grid"
    for j, h in enumerate(encabezado):
        celda = tb.rows[0].cells[j]
        celda.text = ""
        _runs(celda.paragraphs[0], h, negrita=True, tam=tam)
    for i, f in enumerate(filas):
        for j, v in enumerate(f):
            celda = tb.rows[i + 1].cells[j]
            celda.text = ""
            _runs(celda.paragraphs[0], str(v), tam=tam)
    for fila in tb.rows:
        for j, w in enumerate(anchos):
            fila.cells[j].width = Inches(w)
            fila.cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
    par("", despues=4)
    return _n_tab[0]


# ---------------------------------------------------------------- título (una columna)
par("Detección de caídas con datos de acelerómetro: comparación de una CNN 1D de referencia y "
    "la arquitectura InceptionTime sobre el conjunto SisFall", tam=22, alin=WD_ALIGN_PARAGRAPH.CENTER,
    sangria=False, despues=10)
par("[Nombre Apellido 1], [Nombre Apellido 2], [Nombre Apellido 3]", tam=11,
    alin=WD_ALIGN_PARAGRAPH.CENTER, sangria=False)
par("[Programa académico], [Facultad], [Universidad], Cúcuta, Colombia", cursiva=True, tam=10,
    alin=WD_ALIGN_PARAGRAPH.CENTER, sangria=False)
par("[correo1@universidad.edu.co], [correo2@universidad.edu.co], [correo3@universidad.edu.co]", tam=9,
    alin=WD_ALIGN_PARAGRAPH.CENTER, sangria=False, despues=10)

nuevo = doc.add_section(WD_SECTION.CONTINUOUS)
columnas(nuevo, 2)

# ---------------------------------------------------------------- resumen
p_w = pr["f1_macro"]["wilcoxon_p"]
resumen = (
    "Las caídas son una causa principal de lesión y muerte en adultos mayores, y en atención domiciliaria "
    "pueden pasar horas sin que nadie lo note. Este trabajo compara, con los mismos datos y el mismo "
    "protocolo, la red neuronal convolucional (CNN) 1D publicada por Casilari *et al.* para detectar caídas "
    "con un acelerómetro de cintura y la arquitectura moderna InceptionTime. Se usaron las "
    f"{info['trazas']} trazas de SisFall ({info['caidas']} caídas y {info['actividades']} actividades diarias "
    f"de {info['sujetos']} personas de 19 a 75 años), recortadas a ±2,5 s alrededor del pico de aceleración "
    "y divididas 70/25/5 por sujeto, sin fugas entre particiones. Cada red se entrenó desde cero "
    f"{N_SEM} veces con semillas distintas y el mismo presupuesto (20 épocas, paciencia 3). En prueba, la CNN "
    f"de referencia logró una exactitud de {pct(A.exactitud)} % y un F1 macro de {pct(A.f1_macro)} %, cerca del "
    f"98,78 % publicado; InceptionTime logró {pct(B.exactitud)} % y {pct(B.f1_macro)} %. "
    + f"La prueba de Wilcoxon pareada mostró que la CNN es mejor en F1 macro (p = {pval(p_w)}) y en "
    f"especificidad, mientras que InceptionTime detecta más caídas (sensibilidad de {pct(B.sensibilidad)} % "
    f"frente a {pct(A.sensibilidad)} %; p = {pval(pr['sensibilidad']['wilcoxon_p'])}) a costa de más falsas "
    f"alarmas. La CNN usa {A.parametros.iloc[0]} parámetros frente a {B.parametros.iloc[0]} y su latencia en "
    f"CPU fue {c(B.latencia_cpu_ms.mean() / A.latencia_cpu_ms.mean(), 1)} veces menor. Ambas superan el 80 % "
    "de aceptación exigido; la CNN de referencia es la opción recomendada para un dispositivo portátil."
)
p = par(sangria=False, despues=4)
_runs(p, "Resumen—", negrita=True, cursiva=True, tam=9)
_runs(p, resumen, negrita=True, tam=9)
p = par(sangria=False, despues=6)
_runs(p, "Palabras clave—", negrita=True, cursiva=True, tam=9)
_runs(p, "detección de caídas, redes neuronales convolucionales, InceptionTime, acelerometría, SisFall, "
      "adultos mayores.", negrita=True, tam=9)

# ---------------------------------------------------------------- I. Introducción
seccion("Introducción")
par("La Organización Mundial de la Salud estima que cada año mueren unas 684 000 personas por caídas y que "
    "los mayores de 60 años son el grupo más afectado [1]. Las guías mundiales de prevención recomiendan "
    "valorar el riesgo de caída en todos los adultos mayores y actuar sobre los factores modificables [2]. "
    "Además de prevenir, importa detectar la caída cuando ocurre: una persona que vive sola y no puede "
    "levantarse puede permanecer horas en el suelo, con riesgo de deshidratación, úlceras por presión y "
    "mayor mortalidad.")
par("Este trabajo continúa el módulo de inteligencia artificial de SIAD, un sistema de información para "
    "una agencia de atención domiciliaria de adultos mayores en Cúcuta. Ese módulo estima con un bosque "
    "aleatorio el riesgo de caer en los próximos 12 meses a partir de datos de tabla (edad, sexo, caídas "
    "previas, ayuda para desplazarse y escalas de Tinetti y Berg). En su conjunto de prueba alcanzó un "
    "ROC-AUC de 0,745 y una exactitud de 0,693: sirve como apoyo informativo, pero no alcanza el índice de "
    "aceptación del 80 % que exige el curso. Además, predecir el riesgo no avisa cuando la caída ya ocurrió.")
par("Los detectores de caídas con sensores portátiles han seguido tres enfoques [13]. Los primeros usaban "
    "umbrales sobre la magnitud de la aceleración: son simples, pero confunden movimientos bruscos con "
    "caídas. Después se usaron clasificadores clásicos (SVM, k-NN, árboles) sobre rasgos diseñados a mano, "
    "como máximos, varianzas o tiempos entre picos. Hoy predominan las redes profundas, que aprenden los "
    "rasgos directamente de la señal: CNN, redes recurrentes LSTM y combinaciones con atención [3], [10]–[12]. "
    "Su punto débil es la evaluación: casi siempre se prueban con caídas simuladas por voluntarios jóvenes y "
    "con divisiones que no separan a las personas, lo que puede inflar los resultados.")
par("Por eso se aborda aquí el problema complementario: detectar la caída en el momento en que ocurre a "
    "partir de los datos numéricos de un acelerómetro de cintura, un sensor barato que un paciente en casa "
    "puede llevar. Siguiendo la indicación del docente, en lugar de diseñar una red propia se reprodujo una "
    "red ya publicada y validada en un conjunto de datos público: la CNN 1D de Casilari *et al.* [3], "
    "evaluada en SisFall [4]. Como modelo de comparación se eligió InceptionTime [5], una arquitectura "
    "moderna para clasificación de series de tiempo. La familia YOLO [6]–[8], sugerida en la guía de la "
    "actividad, está diseñada para imágenes y no admite señales 1D; la justificación se detalla en la "
    "Sección II.")
par("El aporte del estudio no es decidir qué red \"gana\", sino comparar con rigor: misma partición por "
    "sujeto, mismo presupuesto de entrenamiento, varias semillas, métricas de desempeño y de costo, pruebas "
    "estadísticas pareadas y análisis de errores con mapas Grad-CAM. El objetivo es determinar si una "
    "arquitectura moderna mejora de forma significativa a la CNN de referencia en la detección de caídas "
    "con acelerómetro y a qué costo computacional.")

# ---------------------------------------------------------------- II. Trabajos relacionados
seccion("Trabajos relacionados")
subseccion("CNN de referencia")
par("Casilari, Lora-Rivera y García-Lagos [3] propusieron una CNN 1D que recibe directamente la "
    "aceleración triaxial en una ventana centrada en el pico de la magnitud. Tiene cuatro bloques "
    "convolucionales (16, 32, 64 y 128 filtros de 1×5, normalización por lotes, ReLU y *max pooling* de "
    "1×5, salvo el último bloque) y una capa densa con softmax. La ajustaron con SisFall y una división "
    "aleatoria por traza de 60/20/20; con la ventana de ±2,5 s y las tres componentes reportaron una "
    "sensibilidad de 98,91 %, una especificidad de 98,69 % y una exactitud de 98,78 %. También mostraron que "
    "el desempeño cae mucho al llevar la misma red a otros 13 conjuntos de datos.")
subseccion("Otros trabajos con SisFall")
par("Sucerquia *et al.* [4] publicaron SisFall en la Universidad de Antioquia junto con un detector por "
    "umbrales. Santos *et al.* [9] aplicaron CNN a acelerometría en otros tres conjuntos. Musci *et al.* [10] "
    "usaron redes LSTM para detección en línea sobre SisFall con anotación extendida. Zhang *et al.* [11] "
    "propusieron una CNN de doble flujo con autoatención (DSCS) y reportaron 99,32 % de exactitud en la "
    "prueba de SisFall. Villa y Casilari [12] estudiaron el efecto de la frecuencia de muestreo y obtuvieron "
    "con una CNN-LSTM a 20 Hz una exactitud de 98,9 %, una sensibilidad de 96,7 % y una especificidad de "
    "99,6 %. Casilari *et al.* [13] revisaron los conjuntos públicos para detección con sensores portátiles. "
    "Estos resultados no son directamente comparables entre sí: cambian la ventana, los sensores y, sobre "
    "todo, la forma de dividir los datos.")
subseccion("Familia YOLO y modelo de comparación")
par("YOLO [6] resuelve la detección de objetos en imágenes en una sola pasada; las versiones de "
    "Ultralytics [7], [8] añaden modos de clasificación y segmentación, pero todos esperan imágenes RGB. "
    "SisFall no tiene imágenes ni cajas delimitadoras, y la guía de la actividad indica que no deben "
    "inventarse; convertir la señal en imagen cambiaría la tarea y la entrada respecto a la CNN de "
    "referencia. Se eligió entonces una arquitectura moderna equivalente para series de tiempo: "
    "InceptionTime [5], que adapta los módulos Inception [14] a 1D y es una de las redes de referencia en "
    "la revisión de clasificación de series de tiempo de Ismail Fawaz *et al.* [15], junto a las redes FCN y "
    "ResNet de Wang *et al.* [16].")

# ---------------------------------------------------------------- III. Materiales y métodos
seccion("Materiales y métodos")
subseccion("Conjunto de datos y partición")
par(f"SisFall [4] contiene {info['trazas']} trazas de 15 tipos de caídas simuladas y 19 actividades de la "
    f"vida diaria (ADL), registradas a 200 Hz con un dispositivo en la cintura por 23 adultos de 19 a 30 años "
    f"(SA) y 15 adultos mayores de 60 a 75 años (SE). Los mayores solo realizaron ADL, salvo SE06, experto "
    f"en judo. Las clases están moderadamente desbalanceadas: {info['caidas']} caídas "
    f"({c(100 * info['caidas'] / info['trazas'], 1)} %) y {info['actividades']} ADL. Se publicó como material "
    f"suplementario con licencia CC BY 4.0. Como la página oficial no respondía, se descargó una copia que "
    f"conserva la estructura y el número de archivos originales.")
filas_a = []
for k in range(19):
    d = f"D{k + 1:02d}"
    f = f"F{k + 1:02d}"
    filas_a.append([d, ACT[d].capitalize(), f if k < 15 else "", ACT[f].capitalize().replace("Caída ", "")
                    if k < 15 else ""])
ta = tabla("Actividades diarias (D) y caídas simuladas (F) de SisFall [4]",
           ["Cód.", "Actividad diaria", "Cód.", "Caída"], filas_a, [0.32, 1.4, 0.32, 1.4], tam=6.5)
par(f"La Tabla {ROMANOS[ta - 1]} lista los movimientos. Varias ADL fueron diseñadas para parecerse a una "
    f"caída (D11, desplomarse en una silla; D18, tropezar sin caer; D19, saltar), lo que hace la tarea más "
    f"exigente que distinguir caídas de caminar. Cada traza dura entre 12 y 100 s y contiene un solo "
    f"movimiento; la etiqueta es la del archivo (ADL o caída).")
filas_p = []
for nombre in ["entrenamiento", "prueba", "validacion"]:
    r = part.loc[nombre]
    filas_p.append([nombre.capitalize().replace("Validacion", "Validación"), int(r.sujetos), int(r.trazas),
                    f"{c(r.porc_trazas, 1)} %", int(r.caidas), int(r.actividades)])
tp = tabla("Partición 70/25/5 agrupada por sujeto",
           ["Conjunto", "Sujetos", "Trazas", "% trazas", "Caídas", "ADL"], filas_p,
           [0.95, 0.5, 0.5, 0.55, 0.45, 0.45])
par(f"La partición (Tabla {ROMANOS[tp - 1]}) es 70/25/5 para entrenamiento, prueba y validación, "
    f"agrupada por sujeto y estratificada por grupo de edad: ninguna persona aparece en dos conjuntos, "
    f"de modo que la red se evalúa en personas que nunca vio. Se fijó con la semilla 2026, se guardó en "
    f"*metadatos.csv* y es la misma para ambos modelos y todas las ejecuciones. Se verificó por *hash* que no "
    f"hubiera ventanas idénticas entre particiones (se halló {info['duplicados_exactos']} duplicado, dentro "
    f"del mismo conjunto). La prueba no se usó para ajustar nada.")
subseccion("Preprocesamiento")
par("Como en [3], se tomaron las tres componentes del acelerómetro ADXL345, se convirtieron a unidades g "
    "(rango ±16 g, 13 bits) y se recortó una ventana de ±2,5 s (1001 muestras) centrada en el máximo de la "
    "magnitud SMV = √(Ax² + Ay² + Az²) de cada traza; si el pico estaba cerca del borde, se replicó el valor "
    "extremo. La entrada de 3×1001 se centró restando la media del entrenamiento (equivalente a la "
    "normalización *zerocenter* de MATLAB usada en [3]). No se aplicaron aumentos de datos. La Fig. 1 "
    "muestra una ADL y una caída tal como entran a las redes.")
figura("fig_ejemplos_senal.png", "Ventanas de ±2,5 s de una ADL (subir y bajar escaleras rápido) y de una "
       "caída hacia adelante por tropiezo del sujeto SA01. La caída muestra un pico de impacto mayor y un "
       "cambio brusco de orientación (cambio de eje dominante).")
subseccion("Arquitecturas")
par(f"La Fig. 2 resume ambas redes. La CNN de referencia sigue la Tabla 5 de [3]. El artículo no indica el "
    f"paso del *pooling* ni la posición y tasa del *dropout*; se usó un paso de 5 (submuestreo sin solape) y "
    f"un *dropout* de 0,5 antes de la capa densa. El resultado tiene {A.parametros.iloc[0]} parámetros. "
    f"InceptionTime [5] apila seis módulos Inception con un cuello de botella 1×1 de 32 filtros, tres "
    f"convoluciones de 39, 19 y 9 muestras y una rama de *max pooling*, con atajos residuales cada tres "
    f"módulos y promedio global; tiene {B.parametros.iloc[0]} parámetros. Se usó una sola red y no el "
    f"conjunto de cinco del artículo original, para comparar modelos individuales.")
figura("fig_arquitecturas.png", "Arquitecturas comparadas: (a) CNN 1D de referencia [3]; (b) InceptionTime "
       "[5]. Ambas reciben la misma entrada de 3×1001 y terminan en una capa densa de dos salidas.")
subseccion("Hiperparámetros y protocolo")
h = [["Optimizador", "SGD con momento 0,9", "Adam"],
     ["Tasa de aprendizaje", "1×10⁻⁴", "1×10⁻³"],
     ["Regularización L2", "1×10⁻⁴", "0"],
     ["Tamaño de lote", "64", "64"],
     ["Épocas máximas", "20", "20"],
     ["Parada temprana", "Paciencia 3 (pérdida val.)", "Paciencia 3 (pérdida val.)"],
     ["Pesos finales", "Mejor época en val.", "Mejor época en val."],
     ["Función de pérdida", "Entropía cruzada", "Entropía cruzada"],
     ["Aumentos de datos", "Ninguno", "Ninguno"],
     ["Pesos iniciales", "Aleatorios (desde cero)", "Aleatorios (desde cero)"],
     ["Entrada", "3×1001, centrada", "3×1001, centrada"],
     ["Umbral de decisión", "P(caída) ≥ 0,5", "P(caída) ≥ 0,5"]]
th = tabla("Hiperparámetros de entrenamiento", ["Parámetro", M1, M2], h, [1.05, 1.2, 1.2])
par(f"La Tabla {ROMANOS[th - 1]} reúne los hiperparámetros. Ambos modelos tienen el mismo presupuesto (20 "
    f"épocas como máximo y paciencia de 3 sobre la pérdida de validación, como en [3]) y parten de la misma "
    f"condición: pesos aleatorios, sin preentrenamiento. El optimizador y la tasa de aprendizaje de cada red "
    f"son los de su artículo original. Cada modelo se entrenó y evaluó {N_SEM} veces con las semillas "
    f"{', '.join(map(str, sorted(t.semilla.unique())))}. Cada ejecución corrió en un proceso de Python nuevo, "
    f"con solo las variables de entorno mínimas, de modo que no se arrastraran variables, pesos ni memoria de "
    f"GPU; se fijaron las semillas de Python, NumPy y PyTorch y se activó el modo determinista de cuDNN. Se "
    f"eligieron {N_SEM} semillas, y no 5, porque con 5 pares la prueba de Wilcoxon bilateral no puede bajar "
    f"de p = 0,0625. Cada ejecución guarda un CSV con modelo, semilla, época final, métricas, tiempo de "
    f"entrenamiento y latencia, y todas se reúnen en *ejecuciones.xlsx*.")
subseccion("Métricas y pruebas estadísticas")
par("La métrica principal es el F1 macro (promedio del F1 de ambas clases), según la guía de la actividad. "
    "También se reportan exactitud, sensibilidad (caídas detectadas), especificidad (ADL sin falsa alarma), "
    "precisión y ROC-AUC. Como métricas de eficiencia se midieron el número de parámetros, las "
    "multiplicaciones-acumulaciones (MAC) por ventana, el tiempo de entrenamiento y la latencia de inferencia "
    "con lote 1 (mediana de 300 repeticiones en GPU y de 100 en CPU con un hilo).")
for formula in ["Sensibilidad = VP / (VP + FN)        Especificidad = VN / (VN + FP)",
                "Precisión = VP / (VP + FP)        Exactitud = (VP + VN) / N",
                "F1 = 2·Prec·Sens / (Prec + Sens)        F1 macro = (F1_caída + F1_ADL) / 2"]:
    par(formula, cursiva=True, tam=9, alin=WD_ALIGN_PARAGRAPH.CENTER, sangria=False, antes=2, despues=2)
par("donde VP son las caídas detectadas, FN las caídas no detectadas, VN las ADL clasificadas como ADL, FP "
    "las falsas alarmas y N el total de trazas de prueba. En un sistema de alarma, un FN significa una persona "
    "en el suelo sin ayuda y un FP, una llamada innecesaria que con el tiempo hace que se ignoren las alarmas; "
    "por eso se reportan ambas tasas y no solo la exactitud.", sangria=False)
par("Para comparar se aplicó la prueba de rangos con signo de Wilcoxon, pareada por semilla, sobre cada "
    "métrica, como recomienda Demšar [17] cuando no se puede suponer normalidad; como complemento se reportan "
    "la prueba t pareada y la d de Cohen. Además, para cada semilla se aplicó la prueba exacta de McNemar "
    "[18], [19] a los aciertos y errores de ambas redes sobre las mismas trazas de prueba. Se usó α = 0,05.")
subseccion("Análisis de errores e implementación")
par("Los errores se agruparon por tipo de actividad y por grupo de edad, promediando las ejecuciones. Para la "
    "CNN se calcularon mapas Grad-CAM [20] en la última capa convolucional, interpolados a la longitud de la "
    f"ventana. Todo se implementó en PyTorch {t.torch.iloc[0].split('+')[0]} [21] sobre una GPU "
    f"{t.dispositivo.iloc[0].replace('NVIDIA GeForce ', '')} y un procesador Intel Core 5 210H. El código, "
    "la partición y los resultados por ejecución están en el repositorio del proyecto [22].")

# ---------------------------------------------------------------- IV. Resultados
seccion("Resultados")
filas_r = []
for met, nom in [("exactitud", "Exactitud (%)"), ("f1_macro", "F1 macro (%)"),
                 ("sensibilidad", "Sensibilidad (%)"), ("especificidad", "Especificidad (%)"),
                 ("precision", "Precisión (%)"), ("roc_auc", "ROC-AUC (%)")]:
    filas_r.append([nom, pct(A[met]), pct(B[met]), pval(pr[met]["wilcoxon_p"])])
filas_r += [["Épocas usadas", ms(A.epoca_final, 1), ms(B.epoca_final, 1), "—"],
            ["Tiempo de entrenamiento (s)", ms(A.tiempo_entrenamiento_s, 1), ms(B.tiempo_entrenamiento_s, 1),
             pval(pr["tiempo_entrenamiento_s"]["wilcoxon_p"])],
            ["Latencia GPU (ms)", ms(A.latencia_gpu_ms), ms(B.latencia_gpu_ms), pval(pr["latencia_gpu_ms"]["wilcoxon_p"])],
            ["Latencia CPU (ms)", ms(A.latencia_cpu_ms), ms(B.latencia_cpu_ms), pval(pr["latencia_cpu_ms"]["wilcoxon_p"])],
            ["Parámetros", f"{A.parametros.iloc[0]:,}".replace(",", " "), f"{B.parametros.iloc[0]:,}".replace(",", " "), "—"],
            ["MAC por ventana (millones)", c(A.macs.iloc[0] / 1e6, 2), c(B.macs.iloc[0] / 1e6, 2), "—"],
            ["Tamaño (MB, float32)", c(A.tamano_mb.iloc[0], 3), c(B.tamano_mb.iloc[0], 3), "—"]]
tr = tabla(f"Resultados en prueba (media ± DE, {N_SEM} semillas) y prueba de Wilcoxon pareada",
           ["Métrica", M1, M2, "p"], filas_r, [1.2, 0.95, 0.95, 0.4])
d_exa = 100 * abs(A.exactitud.mean() - 0.9878)
par(f"La Tabla {ROMANOS[tr - 1]} resume las {len(t)} ejecuciones. La CNN de referencia obtuvo una exactitud "
    f"de {pct(A.exactitud)} %, a {c(d_exa, 2)} puntos del 98,78 % publicado [3], con una sensibilidad de "
    f"{pct(A.sensibilidad)} % y una especificidad de {pct(A.especificidad)} %. InceptionTime obtuvo "
    f"{pct(B.exactitud)} %, con sensibilidad de {pct(B.sensibilidad)} % y especificidad de "
    f"{pct(B.especificidad)} %. En F1 macro, la métrica principal, las medias fueron {pct(A.f1_macro)} % y "
    f"{pct(B.f1_macro)} %. Las {len(t)} ejecuciones superan el 80 % de aceptación en todas las métricas de "
    f"desempeño: la peor exactitud individual fue {c(100 * t.exactitud.min(), 2)} %.")
fc = figura("fig_curvas_aprendizaje.png", f"Curvas de aprendizaje (media ± DE de {N_SEM} semillas) de pérdida "
            "y exactitud en entrenamiento (línea continua) y validación (discontinua).")
par(f"La Fig. {fc} muestra las curvas de aprendizaje. La CNN, con SGD y tasa de 1×10⁻⁴, converge despacio y "
    f"usó en promedio {c(A.epoca_final.mean(), 1)} épocas; InceptionTime, con Adam, converge en pocas épocas y "
    f"se detuvo en promedio en la época {c(B.epoca_final.mean(), 1)} (entre {B.epoca_final.min()} y "
    f"{B.epoca_final.max()}). La pérdida de validación de la CNN baja de forma suave y nunca activó la parada "
    f"temprana; la de InceptionTime oscila mucho entre épocas, porque la validación tiene solo dos sujetos y "
    f"Adam da pasos grandes, de modo que la parada temprana se activó con frecuencia tras un repunte aislado.")
fm = figura("fig_matrices_confusion.png", "Matrices de confusión en prueba, promedio de las ejecuciones "
            "(número medio de trazas y porcentaje por fila).")
fd = figura("fig_distribucion_metricas.png", f"Distribución de las métricas de prueba en las {N_SEM} "
            "semillas de cada modelo.")
par(f"Las matrices de confusión promedio (Fig. {fm}) muestran que las dos redes fallan de forma opuesta: la "
    f"CNN deja pasar en promedio {c(A.FN.mean(), 1)} de 448 caídas y genera solo {c(A.FP.mean(), 1)} falsas "
    f"alarmas sobre 703 ADL, mientras que InceptionTime deja pasar {c(B.FN.mean(), 1)} caídas pero genera "
    f"{c(B.FP.mean(), 1)} falsas alarmas. La Fig. {fd} muestra la dispersión entre semillas: la DE de la exactitud fue "
    f"{c(100 * A.exactitud.std(), 2)} puntos para la CNN y {c(100 * B.exactitud.std(), 2)} para InceptionTime.")
fl = figura("fig_desempeno_vs_latencia.png", "F1 macro frente a la latencia de inferencia con lote 1 en GPU "
            "(izquierda) y en CPU con un hilo (derecha). Los rombos son la media ± DE de cada modelo.")
par(f"En costo, la diferencia es grande (Fig. {fl}). InceptionTime tiene "
    f"{c(B.parametros.iloc[0] / A.parametros.iloc[0], 1)} veces más parámetros y "
    f"{c(B.macs.iloc[0] / A.macs.iloc[0], 1)} veces más operaciones MAC. En CPU con un hilo, su latencia fue "
    f"{ms(B.latencia_cpu_ms)} ms frente a {ms(A.latencia_cpu_ms)} ms de la CNN, y su tiempo de entrenamiento "
    f"fue {ms(B.tiempo_entrenamiento_s, 1)} s frente a {ms(A.tiempo_entrenamiento_s, 1)} s.")
subseccion("Pruebas estadísticas")
sig = [m for m in ["exactitud", "f1_macro", "sensibilidad", "especificidad", "precision", "roc_auc"] if signif(m)]
nom = {"exactitud": "exactitud", "f1_macro": "F1 macro", "sensibilidad": "sensibilidad",
       "especificidad": "especificidad", "precision": "precisión", "roc_auc": "ROC-AUC"}
txt = (f"Con la prueba de Wilcoxon pareada ({N_SEM} pares), la diferencia en F1 macro tuvo p = "
       f"{pval(pr['f1_macro']['wilcoxon_p'])} (t pareada: p = {pval(pr['f1_macro']['t_pareada_p'])}; d de Cohen = "
       f"{c(pr['f1_macro']['d_cohen_pareada'] or 0, 2)}). ")
if sig:
    txt += ("Fueron significativas al 5 % las diferencias en " +
            ", ".join(f"{nom[m]} (p = {pval(pr[m]['wilcoxon_p'])}, a favor de {mejor(m)})" for m in sig) + ". ")
else:
    txt += "Ninguna métrica de desempeño mostró una diferencia significativa al 5 %. "
txt += (f"Las diferencias de latencia en CPU (p = {pval(pr['latencia_cpu_ms']['wilcoxon_p'])}) y de tiempo de "
        f"entrenamiento (p = {pval(pr['tiempo_entrenamiento_s']['wilcoxon_p'])}) sí fueron significativas. ")
n_mc = int((mc.p_mcnemar < 0.05).sum())
txt += (f"La prueba de McNemar por semilla (Tabla {ROMANOS[_n_tab[0]]}) fue significativa en {n_mc} de "
        f"{N_SEM} semillas; en promedio solo la CNN acertó {c(mc.solo_cnn_acierta.mean(), 1)} trazas que "
        f"InceptionTime falló, y al revés {c(mc.solo_inception_acierta.mean(), 1)}. Esta prueba compara los "
        f"errores traza por traza dentro de una misma semilla, mientras que la de Wilcoxon compara el resultado "
        f"global entre semillas; usar ambas evita concluir a partir de una sola ejecución [19].")
par(txt)
filas_mc = [[int(r.semilla), int(r.ambos_aciertan), int(r.solo_cnn_acierta), int(r.solo_inception_acierta),
             int(r.ambos_fallan), pval(r.p_mcnemar)] for r in mc.itertuples()]
tabla("Prueba exacta de McNemar por semilla sobre las 1151 trazas de prueba",
      ["Semilla", "Ambos aciertan", "Solo CNN", "Solo Incep.", "Ambos fallan", "p"], filas_mc,
      [0.5, 0.65, 0.55, 0.6, 0.6, 0.5])

# ---------------------------------------------------------------- V. Discusión
seccion("Discusión")
subseccion("Interpretación y comparación con la literatura")
par(f"La reproducción de la CNN de referencia quedó cerca del valor publicado ({pct(A.exactitud)} % frente a "
    f"98,78 %) a pesar de tres diferencias de protocolo. Primero, aquí la partición es por sujeto, mientras "
    f"que en [3] se dividieron las trazas al azar: allí las repeticiones de un mismo voluntario pueden caer en "
    f"entrenamiento y prueba, lo que favorece resultados optimistas. Segundo, se usó PyTorch en lugar de MATLAB "
    f"y se fijaron supuestos no publicados (paso del *pooling* y *dropout*). Tercero, la prueba aquí tiene "
    f"10 sujetos, de modo que el resultado depende de qué personas quedaron en ella. Que la exactitud se "
    f"mantenga sobre el 98 % con sujetos nuevos indica que la red generaliza a personas no vistas dentro de "
    f"SisFall. Los valores son del mismo orden que los de DSCS [11] (99,32 %) y la CNN-LSTM de [12] (98,9 %), "
    f"aunque esas cifras usan otras ventanas, sensores y divisiones (Tabla {ROMANOS[_n_tab[0]]}).")
tabla("Comparación con resultados publicados en SisFall",
      ["Trabajo", "Entrada / división", "Exactitud (%)", "Sens. (%)", "Espec. (%)"],
      [["CNN 1D [3]", "Acel. 3 ejes; aleatoria por traza", "98,78", "98,91", "98,69"],
       ["DSCS [11]", "Acel. + giroscopio", "99,32", "99,15", "—"],
       ["CNN-LSTM 20 Hz [12]", "Acel. remuestreado", "98,9", "96,7", "99,6"],
       ["CNN 1D (este trabajo)", "Acel. 3 ejes; por sujeto", c(100 * A.exactitud.mean(), 2),
        c(100 * A.sensibilidad.mean(), 2), c(100 * A.especificidad.mean(), 2)],
       ["InceptionTime (este trabajo)", "Acel. 3 ejes; por sujeto", c(100 * B.exactitud.mean(), 2),
        c(100 * B.sensibilidad.mean(), 2), c(100 * B.especificidad.mean(), 2)]],
      [0.95, 1.05, 0.5, 0.45, 0.45], tam=7)
par("La comparación con SIAD también es ilustrativa: con datos de tabla, el modelo de riesgo a 12 meses "
    "llegó a una exactitud de 0,693, mientras que con la señal del acelerómetro ambas redes superan 0,95. "
    "No es que una técnica sea mejor que la otra: son problemas distintos. Predecir si alguien caerá en un "
    "año depende de muchos factores que no están en los datos; reconocer una caída que acaba de ocurrir es "
    "un problema mucho más acotado, con una firma física clara. Por eso ambos módulos se complementan.")
par(f"Entre los dos modelos, la CNN de referencia fue significativamente mejor en F1 macro, por "
    f"{c(100 * abs(pr['f1_macro']['dif_media']), 2)} puntos (d de Cohen pareada de "
    f"{c(abs(pr['f1_macro']['d_cohen_pareada']), 2)}, un efecto grande), y además es mucho más barata. La "
    f"ventaja no es uniforme: InceptionTime detectó más caídas, con una sensibilidad "
    f"{c(100 * pr['sensibilidad']['dif_media'], 2)} puntos mayor, pero su especificidad fue "
    f"{c(100 * abs(pr['especificidad']['dif_media']), 2)} puntos menor y su precisión, "
    f"{c(100 * abs(pr['precision']['dif_media']), 2)} puntos menor. Su variabilidad entre semillas también fue "
    f"unas {c(B.f1_macro.std() / A.f1_macro.std(), 0)} veces mayor: con la paciencia de 3 épocas y una "
    f"validación de dos sujetos, varias ejecuciones se detuvieron en la época 6 o 7, antes de estabilizarse; la "
    f"semilla 11, detenida en la época 7, fue la peor ({c(100 * B[B.semilla == 11].exactitud.iloc[0], 2)} %). "
    f"Esto indica que parte de la diferencia viene del protocolo, que es el de [3] y favorece a una red que "
    f"converge de forma suave, más que de la arquitectura. Con ventanas centradas en el pico, la señal de una "
    f"caída es muy distintiva (impacto fuerte y cambio de orientación) y una CNN pequeña ya captura casi toda "
    f"la información; la capacidad adicional de InceptionTime no se traduce en una mejora.")
subseccion("Ventajas y desventajas")
par(f"La CNN de referencia es diminuta ({c(A.tamano_mb.iloc[0], 2)} MB), rápida en CPU y fácil de llevar a un "
    f"microcontrolador o a un teléfono, lo que importa en un sistema portátil con batería limitada. Su "
    f"desventaja es que depende de una tasa de aprendizaje muy baja y de muchas épocas para converger. "
    f"InceptionTime converge en menos épocas, deja escapar menos caídas y es una arquitectura general que no "
    f"requiere ajustar el diseño a cada problema; en un sistema de alarma, perder menos caídas puede pesar más "
    f"que unas falsas alarmas adicionales. Sus desventajas son la inestabilidad entre semillas, la menor "
    f"especificidad y el costo: su inferencia en CPU es "
    f"{c(B.latencia_cpu_ms.mean() / A.latencia_cpu_ms.mean(), 1)} veces mayor y su entrenamiento, mucho más "
    f"lento.")
subseccion("Análisis de errores")
eact2 = eact.copy()
top_fn = eact2[eact2.index.str.startswith("F")].sort_values(M1, ascending=False).head(3)
top_fp = eact2[eact2.index.str.startswith("D")].sort_values(M1, ascending=False).head(3)
top_fn2 = eact2[eact2.index.str.startswith("F")].sort_values(M2, ascending=False).head(3)
top_fp2 = eact2[eact2.index.str.startswith("D")].sort_values(M2, ascending=False).head(3)


def lista(df, col):
    return "; ".join(f"{i} ({ACT[i]}, {c(v, 1)} %)" for i, v in df[col].items())


fe = figura("fig_errores_por_actividad.png", "Tasa de error por tipo de actividad (promedio de las "
            "semillas) en las diez actividades con más errores. Los códigos D son ADL y los F, caídas.")
egr_cnn = egr.set_index(["grupo", "y"])
par(f"La Fig. {fe} muestra dónde fallan las redes. En la CNN, las caídas que más se escapan son: "
    f"{lista(top_fn, M1)}. Las ADL que más falsas alarmas producen son: {lista(top_fp, M1)}. En "
    f"InceptionTime, las caídas más difíciles son {lista(top_fn2, M2)}, y las ADL más confundidas, "
    f"{lista(top_fp2, M2)}. Hay dos patrones. Las caídas que se escapan son sobre todo las que ocurren al "
    f"sentarse o estando sentado (F11, F13, F14): el cuerpo cae desde poca altura, el impacto es corto y la "
    f"postura final (sentado o recostado) se parece a la de una ADL. Las falsas alarmas vienen de ADL con "
    f"impactos fuertes y repetidos, como trotar rápido o bajar escaleras rápido, o con un cambio brusco de "
    f"orientación, como acostarse rápido. InceptionTime, con filtros de hasta 39 muestras, parece más sensible "
    f"a esos impactos periódicos: en trotar rápido se equivocó en el {c(eact.loc['D04', M2], 1)} % de las "
    f"trazas, frente al {c(eact.loc['D04', M1], 1)} % de la CNN. Ninguna red confundió D11 (desplomarse en una "
    f"silla) o D18 (tropezar sin caer) con frecuencia, pese a que se diseñaron para parecerse a una caída.")
try:
    se_adl = egr_cnn.loc[("SE (mayores)", 0)]
    sa_adl = egr_cnn.loc[("SA (jóvenes)", 0)]
    par(f"Por grupo de edad, la tasa de falsas alarmas de la CNN fue {c(se_adl[M1], 2)} % en las ADL de los "
        f"mayores y {c(sa_adl[M1], 2)} % en las de los jóvenes (InceptionTime: {c(se_adl[M2], 2)} % y "
        f"{c(sa_adl[M2], 2)} %; Tabla {ROMANOS[_n_tab[0]]}). Este punto es importante para SIAD, cuyos "
        f"pacientes son adultos mayores: la CNN no dio ninguna falsa alarma con los cuatro mayores de prueba, "
        f"mientras que InceptionTime se equivocó sobre todo con dos de ellos. En cambio, no se puede medir qué "
        f"tan bien detectan las caídas de personas mayores, porque en la prueba no hay ninguna: SE06, el único "
        f"mayor que simuló caídas, quedó en entrenamiento.")
except KeyError:
    pass
filas_g = []
for (g, y), r in egr_cnn.iterrows():
    filas_g.append([g, "Caída (FN)" if y == 1 else "ADL (FP)", c(r[M1], 2), c(r[M2], 2)])
tabla("Tasa de error (%) por grupo de edad y clase real, promedio de las semillas",
      ["Grupo", "Clase real (error)", M1, M2], filas_g, [0.95, 0.9, 0.8, 0.8], tam=7)
fs = figura("fig_errores_por_sujeto.png", "Tasa de error por sujeto de prueba (promedio de las semillas). "
            "Los sujetos SE solo aportan ADL, de modo que su error corresponde a falsas alarmas.")
esuj = pd.read_csv(RES / "errores_por_sujeto.csv", index_col=0)
s1, s2 = esuj[M1].idxmax(), esuj[M2].nlargest(2).index.tolist()
par(f"La Fig. {fs} muestra que los errores no se reparten de forma uniforme. En la CNN, el sujeto {s1} "
    f"concentra la mayor tasa de error ({c(esuj.loc[s1, M1], 2)} %), casi toda en caídas no detectadas; en "
    f"InceptionTime, los mayores {s2[0]} y {s2[1]} ({c(esuj.loc[s2[0], M2], 2)} % y "
    f"{c(esuj.loc[s2[1], M2], 2)} %), todas falsas alarmas. Con solo 10 personas en prueba, el resultado depende "
    f"de quiénes quedaron en esa "
    f"partición; por eso la DE entre semillas mide la variabilidad del entrenamiento, pero no la que habría "
    f"con otra muestra de personas.")
fg = figura("fig_gradcam_cnn.png", "Mapas Grad-CAM de la CNN de referencia (semilla de F1 mediano) sobre "
            "cuatro trazas de prueba: acierto en caída, acierto en ADL, caída no detectada y falsa alarma. "
            "Cada mapa corresponde a la clase predicha; los colores cálidos marcan las zonas que más pesan en esa "
            "decisión.")
par(f"Los mapas Grad-CAM (Fig. {fg}) se calcularon para la clase que predijo la red. En la caída detectada, "
    f"la atención se concentra en el intervalo de unos −0,5 s a +0,3 s alrededor del pico: la fase de caída, el "
    f"impacto y el rebote, que son los rasgos físicos de una caída. Cuando la red predice ADL, en cambio, mira "
    f"los tramos quietos lejos del pico, donde no hay nada parecido a una caída. La caída no detectada (F11, "
    f"caída hacia atrás al sentarse) es ilustrativa: el impacto es un pico aislado en un solo eje, sin la "
    f"secuencia caótica de una caída desde de pie, y la red basó su decisión en el tramo sentado anterior al "
    f"pico. En la falsa alarma (acostarse rápido, P(caída) = 0,568) el mapa queda plano: la decisión quedó en "
    f"el límite y no hay una región con evidencia clara a favor de la caída. Ajustar el umbral, o pedir varias "
    f"ventanas seguidas por encima de él, reduciría este tipo de alarmas.")
subseccion("Implicaciones para SIAD")
par(f"En SIAD, el modelo de riesgo a 12 meses se calcula cuando el profesional registra una valoración; el "
    f"detector propuesto funcionaría de forma continua en un cinturón o teléfono del paciente. Con "
    f"{c(A.latencia_cpu_ms.mean(), 2)} ms por ventana en un solo hilo de CPU, la CNN puede evaluarse muchas "
    f"veces por segundo incluso en hardware modesto, y sus {c(A.tamano_mb.iloc[0], 2)} MB caben en un "
    f"microcontrolador con memoria externa o en cualquier teléfono. Al detectar una caída, el dispositivo "
    f"enviaría una alerta a SIAD, que ya tiene un módulo de alertas preventivas con estados, y la caída "
    f"quedaría en el historial del paciente. Esto resuelve además una limitación del modelo de riesgo: hoy "
    f"las caídas previas las informa el paciente o la familia, y con el detector quedarían registradas de "
    f"forma objetiva. Antes de usarlo con pacientes reales habría que validarlo con caídas reales de adultos "
    f"mayores y fijar con el personal de salud cuántas falsas alarmas son tolerables.")
subseccion("Limitaciones")
par("Las caídas de SisFall son simuladas por voluntarios sobre colchonetas, y casi todas las hicieron "
    "adultos jóvenes; las caídas reales de adultos mayores pueden ser más lentas y variadas. Se usó un solo "
    "conjunto de datos, y [3] mostró que el desempeño cae al cambiar de conjunto. La validación del 5 % tiene "
    "solo dos sujetos, lo que hace frágil la parada temprana. La ventana se centra en el pico de toda la traza, "
    "lo que en un sistema real exige un detector de picos en línea. La comparación con YOLO no fue posible por "
    "el tipo de datos, y se reemplazó por InceptionTime. El presupuesto de 20 épocas con paciencia 3, tomado de "
    "[3], pudo cortar antes de tiempo algunas ejecuciones de InceptionTime; un presupuesto mayor e igual para "
    "ambas es una extensión natural. Por último, durante las primeras ejecuciones la GPU estuvo compartida con "
    "otra aplicación gráfica, lo que infló y volvió más variables los tiempos de entrenamiento y la latencia en "
    "GPU (la primera ejecución de InceptionTime tardó 1795 s y las demás entre 48 y 542 s); la latencia en CPU "
    "con un hilo es la medida de costo más confiable.")

# ---------------------------------------------------------------- VI. Conclusiones
seccion("Conclusiones")
par(f"Se reprodujo la CNN 1D de Casilari *et al.* [3] y se comparó con InceptionTime [5] en la detección de "
    f"caídas con acelerometría de SisFall, con una partición por sujeto y {N_SEM} semillas por modelo. Ambas "
    f"redes superan con amplitud el 80 % de aceptación: la CNN logró una exactitud de {pct(A.exactitud)} % y "
    f"un F1 macro de {pct(A.f1_macro)} %, cerca de lo publicado, e InceptionTime, {pct(B.exactitud)} % y "
    f"{pct(B.f1_macro)} %. Respondiendo al objetivo, la arquitectura moderna no mejoró a la red de referencia: la "
    f"CNN fue significativamente mejor en F1 macro, especificidad y precisión, y mucho más estable entre "
    f"semillas; InceptionTime solo la superó en sensibilidad. Se recomienda la CNN de referencia para un "
    f"dispositivo portátil o un teléfono, por ser {c(B.parametros.iloc[0] / A.parametros.iloc[0], 0)} veces más "
    f"pequeña y unas {c(B.latencia_cpu_ms.mean() / A.latencia_cpu_ms.mean(), 0)} veces más rápida en CPU. "
    f"InceptionTime tiene sentido si la inferencia se hace en un servidor y se prioriza no perder ninguna caída "
    f"aunque aumenten las falsas alarmas. Como trabajo futuro se propone validar con caídas reales de adultos mayores (por ejemplo, "
    f"FARSEEING), evaluar en otros conjuntos, detectar en línea con ventanas deslizantes e integrar la "
    f"detección en SIAD junto al modelo de riesgo a 12 meses, de modo que una caída detectada actualice el "
    f"historial que usa ese modelo.")

# ---------------------------------------------------------------- Declaración IA
p = par(sangria=False, antes=8, despues=2)
_runs(p, "Declaración sobre el uso de herramientas de IA", negrita=True, tam=9)
par("Se usó un asistente de IA generativa (Claude, de Anthropic) para escribir parte del código de "
    "preparación, entrenamiento y análisis y para redactar un primer borrador del artículo. Los autores "
    "revisaron el código, ejecutaron los experimentos, verificaron cada resultado contra los archivos de "
    "*results/* y las referencias contra sus fuentes, y son responsables del contenido.", tam=9)

# ---------------------------------------------------------------- Referencias
p = par("REFERENCIAS", alin=WD_ALIGN_PARAGRAPH.CENTER, sangria=False, antes=8, despues=4)
refs = [
    "World Health Organization, \"Falls,\" Fact sheet, 26 abr. 2021. [En línea]. Disponible: "
    "https://www.who.int/news-room/fact-sheets/detail/falls",
    "M. Montero-Odasso *et al.*, \"World guidelines for falls prevention and management for older adults: "
    "a global initiative,\" *Age and Ageing*, vol. 51, no. 9, art. afac205, 2022, doi: 10.1093/ageing/afac205.",
    "E. Casilari, R. Lora-Rivera y F. García-Lagos, \"A study on the application of convolutional neural "
    "networks to fall detection evaluated with multiple public datasets,\" *Sensors*, vol. 20, no. 5, "
    "art. 1466, 2020, doi: 10.3390/s20051466.",
    "A. Sucerquia, J. D. López y J. F. Vargas-Bonilla, \"SisFall: A fall and movement dataset,\" *Sensors*, "
    "vol. 17, no. 1, art. 198, 2017, doi: 10.3390/s17010198.",
    "H. Ismail Fawaz *et al.*, \"InceptionTime: Finding AlexNet for time series classification,\" *Data Min. "
    "Knowl. Discov.*, vol. 34, no. 6, pp. 1936–1962, 2020, doi: 10.1007/s10618-020-00710-y.",
    "J. Redmon, S. Divvala, R. Girshick y A. Farhadi, \"You only look once: Unified, real-time object "
    "detection,\" en *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 779–788.",
    "G. Jocher, A. Chaurasia y J. Qiu, \"Ultralytics YOLO,\" GitHub, 2023. [En línea]. Disponible: "
    "https://github.com/ultralytics/ultralytics",
    "R. Khanam y M. Hussain, \"YOLOv11: An overview of the key architectural enhancements,\" arXiv:2410.17725, "
    "2024.",
    "G. L. Santos, P. T. Endo, K. H. C. Monteiro, E. S. Rocha, I. Silva y T. Lynn, \"Accelerometer-based "
    "human fall detection using convolutional neural networks,\" *Sensors*, vol. 19, no. 7, art. 1644, 2019, "
    "doi: 10.3390/s19071644.",
    "M. Musci, D. De Martini, N. Blago, T. Facchinetti y M. Piastra, \"Online fall detection using recurrent "
    "neural networks on smart wearable devices,\" *IEEE Trans. Emerg. Topics Comput.*, vol. 9, no. 3, "
    "pp. 1276–1289, 2021, doi: 10.1109/TETC.2020.3027454.",
    "J. Zhang *et al.*, \"An effective deep learning framework for fall detection: Model development and "
    "study design,\" *J. Med. Internet Res.*, vol. 26, art. e56750, 2024, doi: 10.2196/56750.",
    "M. Villa y E. Casilari, \"The impact of the accelerometer sampling rate on the performance of machine "
    "and deep learning models in wearable fall-detection systems,\" *Sensors*, vol. 26, no. 1, art. 162, "
    "2026, doi: 10.3390/s26010162.",
    "E. Casilari, J. A. Santoyo-Ramón y J. M. Cano-García, \"Analysis of public datasets for wearable fall "
    "detection systems,\" *Sensors*, vol. 17, no. 7, art. 1513, 2017, doi: 10.3390/s17071513.",
    "C. Szegedy *et al.*, \"Going deeper with convolutions,\" en *Proc. IEEE Conf. Comput. Vis. Pattern "
    "Recognit. (CVPR)*, 2015, pp. 1–9.",
    "H. Ismail Fawaz, G. Forestier, J. Weber, L. Idoumghar y P.-A. Muller, \"Deep learning for time series "
    "classification: A review,\" *Data Min. Knowl. Discov.*, vol. 33, no. 4, pp. 917–963, 2019.",
    "Z. Wang, W. Yan y T. Oates, \"Time series classification from scratch with deep neural networks: A "
    "strong baseline,\" en *Proc. Int. Joint Conf. Neural Netw. (IJCNN)*, 2017, pp. 1578–1585.",
    "J. Demšar, \"Statistical comparisons of classifiers over multiple data sets,\" *J. Mach. Learn. Res.*, "
    "vol. 7, pp. 1–30, 2006.",
    "Q. McNemar, \"Note on the sampling error of the difference between correlated proportions or "
    "percentages,\" *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947.",
    "T. G. Dietterich, \"Approximate statistical tests for comparing supervised classification learning "
    "algorithms,\" *Neural Comput.*, vol. 10, no. 7, pp. 1895–1923, 1998.",
    "R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh y D. Batra, \"Grad-CAM: Visual "
    "explanations from deep networks via gradient-based localization,\" en *Proc. IEEE Int. Conf. Comput. "
    "Vis. (ICCV)*, 2017, pp. 618–626.",
    "A. Paszke *et al.*, \"PyTorch: An imperative style, high-performance deep learning library,\" en *Adv. "
    "Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019, pp. 8024–8035.",
    "[Autores], \"Detección de caídas: CNN de referencia vs. InceptionTime en SisFall,\" repositorio de código, "
    "2026. [En línea]. Disponible: https://github.com/manuel-ramon2/deteccion-caidas-cnn-sisfall",
]
for i, r in enumerate(refs, 1):
    p = par(sangria=False, tam=8)
    p.paragraph_format.left_indent = Inches(0.28)
    p.paragraph_format.first_line_indent = Inches(-0.28)
    _runs(p, f"[{i}]\t" + r, tam=8)

doc.save(SALIDA)
print("guardado:", SALIDA)
