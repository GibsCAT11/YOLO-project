from ultralytics import YOLO
import os

# Checar si necesito entrar a la carpeta del repo
if 'studienarbeit_repository-main' in os.listdir():
    os.chdir('studienarbeit_repository-main')

# Cargar el modelo base (Nano)
model = YOLO('yolov8n.pt')

print("--- Iniciando Experimento 1: Ajuste de Confianza (0.60) ---")

# Ejecutamos inferencia con el umbral mas alto (0.60)
# Objetivo: Reducir falsos positivos (quitar detecciones basura)
results = model.predict(
    source='prueba.mp4', 
    conf=0.60,        # AQUI ESTA LA MODIFICACION (Default es 0.25)
    device=0,         # Usando la GPU NVIDIA
    save=True,        # Guardar el video resultante para el reporte
    stream=True,      # Importante: Procesa por partes para no llenar la RAM
    show=True         # Mostrar ventana en vivo
)

# Dejar correr el stream frame por frame
for r in results:
    pass 

print("Terminado. Revisa la carpeta 'runs/detect' para el video.")