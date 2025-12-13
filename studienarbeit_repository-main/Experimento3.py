from ultralytics import YOLO
import time
import os

# Verificar carpeta
if 'studienarbeit_repository-main' in os.listdir():
    os.chdir('studienarbeit_repository-main')

def probar_modelo(nombre, video):
    print(f"\n--- 🏁 PROBANDO MODELO: {nombre.upper()} ---")
    
    # Cargar modelo (se descargará solo si no lo tienes)
    model = YOLO(nombre)
    
    # Medir tiempo de inicio
    start = time.time()
    
    # Corremos 200 cuadros para medir velocidad
    # save=True para que puedas ver si corrige los errores de "trenes"
    results = model.predict(source=video, device=0, stream=True, save=True, conf=0.40)
    
    count = 0
    for r in results:
        count += 1
        if count >= 200: break # Cortamos a los 200 frames
            
    total_time = time.time() - start
    fps = count / total_time
    
    print(f"   ⏱️ Tiempo: {total_time:.2f}s | Velocidad: {fps:.2f} FPS")
    return fps

print("--- 🧪 INICIANDO EXPERIMENTO 3: NANO VS SMALL ---")

# 1. Correr el Nano (El que ya conoces)
fps_nano = probar_modelo('yolov8n.pt', 'prueba.mp4')

# 2. Correr el Small (El nuevo, más inteligente)
# Fíjate si este ya no confunde los edificios
fps_small = probar_modelo('yolov8s.pt', 'prueba.mp4')

print(f"\n📊 CONCLUSIÓN:")
print(f"Diferencia de velocidad: El Nano es {fps_nano - fps_small:.2f} FPS más rápido.")