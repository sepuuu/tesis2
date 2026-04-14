# ⚽️ Análisis de Video de Fútbol usando Visión por Computadora 🧠📊

¡Bienvenido al proyecto de **Obtención de mapa de pases utilizando Visión por Computadora**! Este proyecto de título aplica técnicas de visión por computadora para analizar partidos de fútbol: detecta jugadores y balón, realiza re-identificación (ReID), proyecta a un **plano de cancha** y construye un **radar con Voronoi** **estable**. Además, genera artefactos para análisis táctico: posiciones, métricas y (opcional) mapas de pases.

---

## 🚀 Estado actual (septiembre 2025)

- **Entrada multicámara (2 cámaras)** con homografías independientes.
- **Fusión posicional + ReID** (evita duplicados entre cámaras).
- **Tracker global + Roster (IDs 1..11)** con confirmación y reciclaje.
- **Radar + Voronoi** con:
  - **Contornos visibles** y **sitios con anillo**.
  - **Anti-parpadeo** (puntos “pegados” ~1s) para estabilidad al perder detecciones breves.
- **Salida única en MP4** (side-by-side de ambas cámaras con radar/Veronoi incrustado en Cam1).
- **Export a Excel** de posiciones por frame (jugadores + balón).
- **Métricas por frame** (FPS, detecciones, fusión, tracks, dibujados, puntos “pegados”, duplicados, ID switches) con resumen.

![Ejemplo del análisis de video](docs/Imagen_programa.png)

---

## 🎯 Objetivos

Proveer un **pipeline robusto** para análisis táctico con:
- Seguimiento multicámara y proyección a cancha.
- Radar con Voronoi **legible y estable**.
- Artefactos reproducibles: **MP4 único** + **Excel** + **métricas**.
- (Opcional) generación de **mapas de pases**.

---

## 🛠️ Tecnologías utilizadas
- **YOLO** (export ONNX) para detección (jugadores y balón).
- **TorchReID** para embeddings de re-identificación.
- **Decord** para lectura de video, **OpenCV** para homografía/Voronoi y composición.
- **Numpy/Pandas** para datos y export a Excel.
- Proyecto probado en **Python 3.11**.

---

## 🧪 Resultados y artefactos
Al ejecutar el pipeline se generan:
- **Video**: `codes/outputs/Pruebas-output-final.mp4` (side-by-side; radar+Voronoi en Cam1).
- **Excel**: `codes/data/Posiciones-jugadores-balon-multicam.xlsx` (Frame, Id, PosX, PosY, BallX, BallY).
- **Métricas**: `codes/data/METRICS.csv` (por frame) y `codes/data/METRICS_SUMMARY.txt` (resumen).

**Ejemplo de Mapa de Pases (opcional):**  
[🔗 Ver Mapa de Pases (PDF)](docs/mapa_pases.pdf)

---

## 📦 Preparación de entorno

```bash
# Python 3.11 recomendado
pip install -r requirements.txt
```

Estructura de carpetas recomendada:
```
codes/
  inputs/                       # Videos de entrada (Cam1, Cam2)
  outputs/                      # Salidas (MP4)
  models/                       # Pesos (players.onnx, ball.onnx, model.pth.tar-300)
  data/                         # Artefactos (Excel, métricas)
configs/
  drawing.py
  soccer.py
  view_transformer.py
utils/
  drawing_utils.py
  ball_setup.py
  tracking.py
  team.py
  train_team_classifier.py
main_dualcam.py
```

---

## ▶️ Cómo usar el proyecto

1. Asegúrate de tener los modelos en `codes/models/`:
   - `players.onnx` (jugadores)
   - `ball.onnx` (balón)
   - `model.pth.tar-300` (ReID)

2. Coloca los videos en `codes/inputs/`:
   - `video_blancos_corto.mp4` (Cam1)
   - `video_negros_corto.mp4` (Cam2)

3. Ejecuta:
```bash
python main_dualcam.py
```

4. Revisa los resultados:
   - Video anotado: `codes/outputs/Pruebas-output-final.mp4`
   - Excel: `codes/data/Posiciones-jugadores-balon-multicam.xlsx`
   - Métricas: `codes/data/METRICS.csv`, `codes/data/METRICS_SUMMARY.txt`

---

## 📂 Archivos Faltantes

El proyecto requiere algunos archivos adicionales que no están incluidos directamente en el repositorio debido a su tamaño. A continuación, se listan los archivos necesarios junto con sus enlaces de descarga y su ubicación esperada dentro del proyecto:

### 1. **Modelo de Re-ID**
Este modelo es esencial para la reidentificación de los jugadores durante el análisis de los videos.

**Descarga aquí**: [🔗 Descargar Modelo de Re-ID](https://drive.google.com/file/d/1WUUdcJ29A11i1zoipnq7mqQZeR84V_PV/view?usp=sharing)

**Ubicación esperada:**  
Coloca este archivo en la carpeta `models/` (crea esta carpeta si no existe).

### 2. **Video para Pruebas**
Utiliza este video para probar el pipeline completo del proyecto.

**Descarga aquí**: [🔗 Descargar Video de Prueba](https://drive.google.com/file/d/1vVypn9X0mfgurgtj4fmnpGnsMn8SDMpw/view?usp=drive_link)

**Ubicación esperada:**  
Coloca este archivo en la carpeta `videos/` (crea esta carpeta si no existe).

---

## ⚠️ Estado del proyecto

El proyecto está en desarrollo. Las **instrucciones detalladas** de configuración y los **scripts de generación de mapas de pases** se añadirán más adelante; de momento, sigue los pasos indicados en este README.

---

## 👥 Autores

- **Matías Millacura** - [@matiasmillacura](https://github.com/matiasmillacura)
- **Matías Sepúlveda** - [@sepuuu](https://github.com/sepuuu)
---

## Configuracion ajustable

Los archivos de la carpeta `codes/configs/` concentran los parametros mas comunes que vas a querer ajustar antes de correr `main_dualcam.py`. Aprovecha el siguiente resumen para saber donde tocar cada cosa:

| Archivo | Ruta | Para que sirve | Parametros que puedes ajustar |
| --- | --- | --- | --- |
| `soccer.py` | `codes/configs/soccer.py` | Define la geometria base de la cancha (largo, ancho y vertices usados para dibujar las lineas). | Modifica `width`, `length`, los campos `penalty_box_*` y las listas `vertices` / `edges` si trabajas con otra escala o layout. |
| `view_transformer.py` | `codes/configs/view_transformer.py` | Calcula la homografia imagen->cancha. | Ajusta en `main_dualcam.py` los arrays `points_img*` y `points_pitch*` para cada camara; esta clase aplica la transformacion resultante. |
| `drawing.py` | `codes/configs/drawing.py` | Renderiza el radar y las capas Voronoi. | En `PitchRenderer` puedes cambiar `scale`, `padding`, `background_color`, `line_color` para adaptar el mini mapa. |
| `team.py` | `codes/configs/team.py` | Clasificador de equipacion (SigLIP + UMAP + KMeans) y logica de locking. | Tunea `use_umap`, `umap_components`, `vote_window`, `tight_lock_p`, `low_conf_thr` o usa `manual_lock` para fijar resultados manuales. |
| `ball.py` | `codes/configs/ball.py` | Buffers, tracker y anotador de la trayectoria del balon. | Ajusta `buffer_size`, `max_radius`, `thickness` para suavizar o resaltar la estela del balon. |

> Sugerencia: despues de tocar la configuracion limpia o regenera los artefactos en `codes/data/` y vuelve a ejecutar `python codes/main_dualcam.py` para validar el efecto.

