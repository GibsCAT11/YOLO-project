from ultralytics import YOLO
import os

# Verificamos si necesitamos entrar a la carpeta del repositorio
if 'studienarbeit_repository-main' in os.listdir():
    os.chdir('studienarbeit_repository-main')

# Cargamos el modelo base
model = YOLO('yolov8n.pt')

print("--- 🧪 INICIANDO EXPERIMENTO 2: AJUSTE DE NMS (IOU=0.3) ---")
print("Objetivo: Eliminar detecciones duplicadas (cajas encimadas).")

# Ejecutamos la inferencia
# iou=0.3: Si dos cajas se superponen más del 30%, borra una.
# conf=0.25: Mantenemos la confianza estándar para aislar el efecto del NMS.
results = model.predict(
    source='prueba.mp4', 
    iou=0.3,          # <--- MODIFICACIÓN CLAVE (Default es 0.7)
    conf=0.25,        # Confianza normal
    device='cpu',         # Usando tu GTX 1650
    save=True,        # Guardamos el video para evidencia
    stream=True,      # Modo seguro para RAM
    show=True         # Ver en vivo
)

# Bucle para procesar el video cuadro por cuadro
for r in results:
    pass 

print("✅ Experimento 2 terminado. Video guardado en 'runs/detect'.")