import cv2
from ultralytics import YOLO
import time
import argparse
import os
import sys
import matplotlib.pyplot as plt
from fpdf import FPDF

# --- CONFIGURACIÓN DE ARGUMENTOS ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='yolov8n.pt', help='Modelo a usar')
parser.add_argument('--conf', type=float, default=0.25, help='Confianza')
parser.add_argument('--iou', type=float, default=0.7, help='NMS')
parser.add_argument('--imgsz', type=int, default=640, help='Resolución')
parser.add_argument('--source', type=str, default='prueba.mp4', help='Video source')
parser.add_argument('--report', action='store_true', help='Generar PDF al finalizar')
args = parser.parse_args()

# --- FUNCIONES DE REPORTE ---
def generar_grafica(stats):
    try:
        clases = list(stats['class_counts'].keys())
        conteos = list(stats['class_counts'].values())
        
        plt.figure(figsize=(6, 4))
        plt.bar(clases, conteos, color=['#0077b6', '#00b4d8', '#90e0ef', '#caf0f8'])
        plt.title('Objetos Detectados en la Sesión')
        plt.xlabel('Clase')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("temp_graph.png")
        plt.close()
        return True
    except:
        return False

def crear_pdf(stats, video_name):
    print("\n📄 Generando Reporte PDF...")
    pdf = FPDF()
    pdf.add_page()
    
    # Encabezado
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Reporte de Analisis: {video_name}", ln=1, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "Generado por Gibran IA Suite", ln=1, align='C')
    pdf.ln(10)
    
    # Resumen
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Resumen de Ejecucion", ln=1)
    pdf.set_font("Arial", '', 11)
    
    texto_resumen = (
        f"Duracion total: {stats['total_time']:.2f} segundos\n"
        f"Total frames procesados: {stats['total_frames']}\n"
        f"Velocidad promedio: {stats['avg_fps']:.2f} FPS\n"
        f"Confianza promedio de deteccion: {stats['avg_conf']:.2f}%"
    )
    pdf.multi_cell(0, 10, texto_resumen)
    pdf.ln(5)
    
    # Gráfica
    if generar_grafica(stats):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "2. Distribucion de Objetos", ln=1)
        pdf.image("temp_graph.png", x=15, w=180)
        if os.path.exists("temp_graph.png"): os.remove("temp_graph.png")
    
    nombre_pdf = f"Reporte_{video_name.split('.')[0]}.pdf"
    try:
        pdf.output(nombre_pdf)
        print(f"✅ PDF Guardado: {nombre_pdf}")
    except Exception as e:
        print(f"❌ Error guardando PDF: {e}")

# --- REPRODUCTOR PRINCIPAL ---
def run_player():
    # Resolver ruta del video
    if os.path.isabs(args.source):
        ruta_video = args.source
    else:
        ruta_video = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.source)

    if not os.path.exists(ruta_video):
        print(f"❌ Error: No existe {ruta_video}")
        return

    print(f"🚀 Iniciando análisis de: {os.path.basename(ruta_video)}")
    model = YOLO(args.model)
    cap = cv2.VideoCapture(ruta_video)
    
    # Ventana
    window_name = f"Gibran IA - {os.path.basename(ruta_video)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 600)

    # Variables Estadísticas
    stats = {
        'total_frames': 0,
        'class_counts': {},
        'conf_sum': 0,
        'detections': 0
    }
    
    start_time = time.time()
    paused = False
    delay = 1

    try:
        while cap.isOpened():
            if paused:
                key = cv2.waitKey(100)
                if key == 32: paused = not paused
                elif key == 27: break
                continue
            
            success, frame = cap.read()
            if not success: break

            stats['total_frames'] += 1

            # Inferencia
            results = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
            annotated_frame = results[0].plot()

            # Recolectar datos en vivo
            if args.report:
                for r in results:
                    for box in r.boxes:
                        cls_name = model.names[int(box.cls[0])]
                        stats['class_counts'][cls_name] = stats['class_counts'].get(cls_name, 0) + 1
                        stats['conf_sum'] += float(box.conf[0])
                        stats['detections'] += 1

            cv2.imshow(window_name, annotated_frame)

            # Controles
            key = cv2.waitKey(delay) & 0xFF
            if key == 27: break # ESC
            elif key == 32: paused = not paused # Espacio
            elif key == ord('w'): delay = max(1, delay - 5)
            elif key == ord('s'): delay += 5

    except Exception as e:
        print(f"❌ Interrupción: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Cálculos Finales
    total_time = time.time() - start_time
    stats['total_time'] = total_time
    stats['avg_fps'] = stats['total_frames'] / total_time if total_time > 0 else 0
    stats['avg_conf'] = (stats['conf_sum'] / stats['detections'] * 100) if stats['detections'] > 0 else 0

    print(f"\n🏁 Fin de sesión. {stats['total_frames']} frames procesados.")
    
    # Generar Reporte si fue solicitado
    if args.report:
        crear_pdf(stats, os.path.basename(ruta_video))

if __name__ == "__main__":
    run_player()