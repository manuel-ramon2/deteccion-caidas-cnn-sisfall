# Detección de caídas con datos de sensores: CNN de referencia vs. InceptionTime

Actividad final del curso: análisis comparativo de una CNN publicada frente a un
modelo moderno, con los mismos datos y el mismo protocolo. Es la continuación
del módulo de IA de **SIAD**, que estima el riesgo de caída a 12 meses con datos
de tabla. Aquí el problema es complementario: **detectar si una caída ocurrió**
a partir de los datos numéricos de un acelerómetro de cintura.

| | |
|---|---|
| Tarea | Clasificación binaria de series de tiempo: caída vs. actividad diaria (ADL) |
| Datos | SisFall (Sucerquia et al., 2017; Universidad de Antioquia, CC BY 4.0), 4.505 trazas, 38 sujetos |
| Línea base | CNN 1D de Casilari, Lora-Rivera y García-Lagos (Sensors, 2020) |
| Comparación | InceptionTime (Ismail Fawaz et al., 2020). YOLO no aplica a señales 1D (ver artículo) |
| Métrica principal | F1 macro (más exactitud, sensibilidad, especificidad, ROC-AUC, latencia, parámetros, MACs) |

## Estructura

```
data/
  preparar_datos.py     ventanas ±2,5 s alrededor del pico y partición 70/25/5 por sujeto
  procesado/            X.npy, metadatos.csv (con la partición), resumen_particion.csv, info.json
  raw/SisFall/          dataset original (se descarga, no se versiona)
src/
  modelos.py            CNN de referencia e InceptionTime
  entrenar.py           una ejecución (modelo + semilla) en un proceso nuevo
  ejecutar_todo.py      10 semillas por modelo, entorno limpio entre ejecuciones
  analizar.py           tabla media ± DE, Wilcoxon, t pareada, McNemar, figuras, errores y Grad-CAM
results/
  ejecuciones/          un CSV por ejecución, historia por época, predicciones y pesos
  ejecuciones.xlsx      todas las ejecuciones (modelo, semilla, época final, métricas, tiempo, latencia)
  figuras/              figuras del artículo
articulo/               artículo en formato IEEE
```

## Cómo correr todo

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 1. Descargar SisFall (la página oficial de la UdeA no responde; copia con la estructura original)
git clone --depth 1 https://github.com/Fall-Prevention-Team/sisfallData.git data/raw/SisFall

# 2. Ventanas y partición (fija: semilla 2026)
python data/preparar_datos.py --raw data/raw/SisFall --out data/procesado

# 3. Entrenar y evaluar 2 modelos × 10 semillas (~15-25 min con GPU)
python src/ejecutar_todo.py

# 4. Estadística, figuras, análisis de errores y Grad-CAM
python src/analizar.py
```

`ejecutar_todo.py` salta las ejecuciones que ya tienen su CSV; para repetir desde
cero, borre `results/ejecuciones/`.

## Protocolo (resumen)

- Misma partición para ambos modelos: 70/25/5 (entrenamiento/prueba/validación)
  **agrupada por sujeto** y estratificada por grupo de edad. Ningún sujeto ni ventana
  idéntica aparece en dos particiones (`data/procesado/info.json`).
- Mismo presupuesto: máximo 20 épocas, parada temprana con paciencia 3 sobre la
  pérdida de validación, lote 64, se restauran los pesos de la mejor época.
- Ambos modelos parten de cero (sin pesos preentrenados) y reciben la misma entrada:
  aceleración triaxial en g centrada con la media del entrenamiento.
- La prueba (25 %) no se usa para ajustar nada.
- 10 semillas por modelo, cada una en un proceso de Python nuevo con variables de
  entorno mínimas.

## Uso de los datos

SisFall se publicó como material suplementario de Sucerquia et al. (2017) con
licencia CC BY 4.0. Las caídas son simuladas por voluntarios sobre colchonetas;
los adultos mayores (SE) solo realizaron actividades diarias, salvo SE06.
