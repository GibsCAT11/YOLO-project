import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os

# --- CONFIGURACIÓN DE LA VENTANA ---
def crear_interfaz():
    ventana = tk.Tk()
    ventana.title("Sistema de Detección Vehicular - YOLOv8")
    ventana.geometry("600x550")
    ventana.configure(bg="#f0f0f0") # Fondo gris claro profesional

    # Estilo visual
    estilo = ttk.Style()
    estilo.theme_use('clam') # Un tema más moderno
    estilo.configure("TButton", font=("Segoe UI", 10), padding=10)
    estilo.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 11))
    
    # --- FUNCIONES ---
    def ejecutar_script(nombre_archivo):
        """Lanza el script del experimento en una terminal separada"""
        if not os.path.exists(nombre_archivo):
            messagebox.showerror("Error", f"No encuentro el archivo: {nombre_archivo}\nVerifica que esté en la carpeta.")
            return

        lbl_estado.config(text=f"⏳ Ejecutando: {nombre_archivo}...", fg="blue")
        ventana.update()
        
        try:
            # subprocess.run abre el archivo como si escribieras 'python archivo.py' en la terminal
            subprocess.run(["python", nombre_archivo], check=True)
            lbl_estado.config(text=f"✅ Finalizado: {nombre_archivo}", fg="green")
        except Exception as e:
            lbl_estado.config(text="❌ Error en la ejecución", fg="red")
            messagebox.showerror("Error de Ejecución", str(e))

    # --- ELEMENTOS DE LA INTERFAZ (WIDGETS) ---

    # 1. Título Principal
    lbl_titulo = tk.Label(ventana, text="🚗 Proyecto YOLOv8", bg="#f0f0f0", 
                          font=("Segoe UI", 20, "bold"), fg="#333")
    lbl_titulo.pack(pady=(20, 5))

    # 2. Subtítulo / Autor
    lbl_subtitulo = tk.Label(ventana, text="Detección de Objetos para Conducción Autónoma", 
                             bg="#f0f0f0", font=("Segoe UI", 12), fg="#666")
    lbl_subtitulo.pack(pady=(0, 20))

    # 3. Descripción del Proyecto
    descripcion = ("Este software utiliza Visión por Computadora e Inteligencia Artificial "
                   "para detectar vehículos, peatones y señales en tiempo real.\n\n"
                   "Seleccione un experimento para visualizar los resultados:")
    
    lbl_desc = tk.Label(ventana, text=descripcion, bg="#ffffff", fg="#444",
                        justify="left", wraplength=500, padx=20, pady=15, 
                        relief="groove", font=("Segoe UI", 10))
    lbl_desc.pack(pady=10)

    # 4. Panel de Botones (Portafolio)
    frame_botones = tk.Frame(ventana, bg="#f0f0f0")
    frame_botones.pack(pady=20)

    # Botón Exp 1
    btn_exp1 = ttk.Button(frame_botones, text="🧪 Exp 1: Sensibilidad (Confianza)", 
                          command=lambda: ejecutar_script("Experimento1.py"))
    btn_exp1.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

    # Botón Exp 2
    btn_exp2 = ttk.Button(frame_botones, text="🧪 Exp 2: Limpieza (NMS/IoU)", 
                          command=lambda: ejecutar_script("Experimento2.py"))
    btn_exp2.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

    # Botón Exp 3
    btn_exp3 = ttk.Button(frame_botones, text="🏎️ Exp 3: Velocidad (Nano vs Small)", 
                          command=lambda: ejecutar_script("Experimento3.py"))
    btn_exp3.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

    # Botón Exp 4
    btn_exp4 = ttk.Button(frame_botones, text="📏 Exp 4: Resolución y Cold Start", 
                          command=lambda: ejecutar_script("Experimento4.py"))
    btn_exp4.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

    # 5. Barra de Estado
    lbl_estado = tk.Label(ventana, text="Listo para iniciar.", bg="#e0e0e0", 
                          font=("Consolas", 9), anchor="w")
    lbl_estado.pack(side="bottom", fill="x")

    # Botón Salir
    btn_salir = tk.Button(ventana, text="Salir", command=ventana.quit, 
                          bg="#ffcccc", font=("Segoe UI", 9))
    btn_salir.place(x=520, y=510)

    ventana.mainloop()

if __name__ == "__main__":
    crear_interfaz()