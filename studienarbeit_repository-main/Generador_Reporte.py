import sys
import os
import cv2
import time
import argparse
import matplotlib.pyplot as plt
from ultralytics import YOLO
from fpdf import FPDF

# --- CONFIGURACIÓN ---
parser = argparse.ArgumentParser()
parser.add_argument("video_path", help="Ruta del video a analizar")
args = parser.parse_args()

def generar_graficas(stats):
    """Crea gráficas temporales para el reporte"""
    # 1. Gráfica de Clases Detectadas
    clases = list(stats['class_counts'].keys())
    conteos = list(stats['class_counts'].values())
    
    plt.figure(figsize=(6, 4))
    plt.bar(clases, conteos, color=['#0077b6', '#00b4d8', '#90e0ef', '#caf0f8'])
    plt.title('Distribución de Objetos Detectados')
    plt.xlabel('Tipo de Objeto')
    plt.ylabel('Cantidad Total (Apariciones)')
    plt.tight_layout()
    plt.savefig("temp_graph_bar.png")
    plt.close()

def crear_pdf(stats, output_filename="Reporte_Final_YOLO.pdf"):
    pdf = FPDF()
    pdf.add_page()
    
    # Título
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Reporte Técnico de Análisis Vehicular", ln=1, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt="Generado por: YOLOv8 Engineering Suite", ln=1, align='C')
    
    pdf.ln(10)
    
    # Resumen Ejecutivo
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="1. Resumen Ejecutivo", ln=1, align='L')
    pdf.set_font("Arial", '', 11)
    
    resumen = (
        f"Se realizo un analisis completo sobre el video proporcionado.\n"
        f"Duracion del analisis: {stats['total_time']:.2f} segundos.\n"
        f"Total de cuadros (frames) procesados: {stats['total_frames']}.\n"
        f"Velocidad promedio de procesamiento: {stats['avg_fps']:.2f} FPS.\n"
    )
    pdf.multi_cell(0, 10, txt=resumen)
    
    pdf.ln(5)
    
    # Detalles de Detección
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="2. Estadísticas de Detección", ln=1, align='L')
    pdf.set_font("Arial", '', 11)
    
    detalles = (
        f"Modelo utilizado: YOLOv8s (Small - Alta Precision)\n"
        f"Confianza Promedio: {stats['avg_conf']:.2f}%\n"
        f"Total de objetos unicos rastreados: {sum(stats['class_counts'].values())} (aprox)\n"
    )
    pdf.multi_cell(0, 10, txt=detalles)
    
    pdf.ln(5)

    # Insertar Gráfica
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="3. Visualización de Datos", ln=1, align='L')
    pdf.image("temp_graph_bar.png", x=10, w=190)
    
    # Guardar
    pdf.output(output_filename)
    print(f"📄 PDF Generado exitosamente: {output_filename}")
    
    # Limpiar
    if os.path.exists("temp_graph_bar.png"):
        os.remove("temp_graph_bar.png")

def analizar_video():
    video_path = args.video_path
    if not os.path.exists(video_path):
        print("❌ Error: Archivo no encontrado.")
        return

    print(f"⏳ Iniciando Análisis Profundo en: {os.path.basename(video_path)}")
    print("   Esto puede tardar dependiendo de la duración del video...")

    # Usamos el modelo SMALL para mejor reporte (combina exp 3)
    model = YOLO('yolov8s.pt') 
    cap = cv2.VideoCapture(video_path)
    
    stats = {
        'total_frames': 0,
        'class_counts': {},
        'conf_sum': 0,
        'detections_count': 0
    }
    
    start_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        stats['total_frames'] += 1
        
        # Inferencia silenciosa (sin video en pantalla para máxima velocidad)
        # Usamos iou=0.3 (Exp 2) y conf=0.4 (Balanceado)
        results = model.predict(frame, conf=0.4, iou=0.3, verbose=False)
        
        # Recolectar datos
        for r in results:
            for box in r.boxes:
                # Contar clases
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                stats['class_counts'][cls_name] = stats['class_counts'].get(cls_name, 0) + 1
                
                # Promedio confianza
                stats['conf_sum'] += float(box.conf[0])
                stats['detections_count'] += 1

        # Barra de progreso simple en consola
        if stats['total_frames'] % 50 == 0:
            print(f"   -> Procesando frame {stats['total_frames']}...")

    cap.release()
    end_time = time.time()
    
    # Cálculos finales
    stats['total_time'] = end_time - start_time
    stats['avg_fps'] = stats['total_frames'] / stats['total_time']
    if stats['detections_count'] > 0:
        stats['avg_conf'] = (stats['conf_sum'] / stats['detections_count']) * 100
    else:
        stats['avg_conf'] = 0

    print("\n✅ Análisis completado. Generando PDF...")
    generar_graficas(stats)
    crear_pdf(stats)

if __name__ == "__main__":
    analizar_video()