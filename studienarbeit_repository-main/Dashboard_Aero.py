import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from PIL import Image, ImageTk
import subprocess
import threading
import os
import sys

# --- ESTILO ---
FONT_TITLE = ("Segoe UI", 16, "bold")
COLOR_GLASS = "#dceefc"
COLOR_TXT_INFO = "#023e8a"

class AeroDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Engineering Suite - Gibran IA")
        
        # Geometría centrada
        ancho, alto = 1050, 720
        x_pos = (root.winfo_screenwidth() // 2) - (ancho // 2)
        y_pos = (root.winfo_screenheight() // 2) - (alto // 2)
        self.root.geometry(f"{ancho}x{alto}+{x_pos}+{y_pos}")
        self.root.resizable(False, False)
        
        self.current_process = None

        # Fondo
        try:
            self.bg_image = Image.open("fondo_aero.png")
            self.bg_image = self.bg_image.resize((ancho, alto), Image.Resampling.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)
            tk.Label(root, image=self.bg_photo).place(x=0, y=0, relwidth=1, relheight=1)
        except:
            self.root.configure(bg="#8ecae6")

        # === PANEL IZQUIERDO ===
        self.frame_left = tk.Frame(root, bg="white", bd=3, relief="ridge")
        self.frame_left.place(x=20, y=20, width=320, height=alto - 40)
        
        tk.Label(self.frame_left, text="Panel de Control", font=FONT_TITLE, bg="white", fg="#0077b6").pack(pady=(20, 5))

        # Botones Experimentos 
        self.crear_boton("🧪 Exp 1: Sensibilidad", [sys.executable, "Smart_Player.py", "--conf", "0.60"], "Alta confianza (0.60).")
        self.crear_boton("🔎 Exp 2: Limpieza NMS", [sys.executable, "Smart_Player.py", "--iou", "0.3"], "IoU estricto (0.3).")
        self.crear_boton("🏎️ Exp 3: Modelo Small", [sys.executable, "Smart_Player.py", "--model", "yolov8s.pt"], "Modelo preciso (Small).")
        ttk.Separator(self.frame_left, orient='horizontal').pack(fill='x', pady=15, padx=10)

        # --- ZONA DE VIDEO PROPIO (EL BOTÓN MAESTRO) ---
        tk.Label(self.frame_left, text="Análisis Completo", font=("Segoe UI", 10, "bold"), bg="white", fg="#333").pack()
        
        #  Btn Carga y analisís video 
        btn_master = tk.Button(self.frame_left, text="📂 CARGAR VIDEO + REPORTE", 
                             bg="#ffd700", fg="#333", cursor="hand2", font=("Segoe UI", 10, "bold"), relief="raised",
                             command=self.analisis_completo)
        btn_master.pack(pady=10, padx=20, fill="x")
        
        # Info Dinámica
        self.lbl_info_texto = tk.Label(self.frame_left, text="Selecciona una opción.", 
                                       font=("Segoe UI", 9), bg="#f0f8ff", justify="left", wraplength=280, relief="groove", bd=1, padx=10, pady=10)
        self.lbl_info_texto.pack(side="bottom", pady=20, padx=15, fill="x")

        # Botón Detener
        self.btn_stop = tk.Button(self.frame_left, text="⛔ DETENER PROCESO", 
                                  font=("Segoe UI", 10, "bold"), bg="#ffcccc", fg="red",
                                  state="disabled", cursor="hand2", command=self.detener_proceso)
        self.btn_stop.pack(side="bottom", pady=10, padx=20, fill="x")

        # === PANEL DERECHO ===
        self.frame_right = tk.Frame(root, bg=COLOR_GLASS, bd=0)
        self.frame_right.place(x=360, y=20, width=ancho - 380, height=alto - 40)

        # Instrucciones
        self.frame_instr = tk.Frame(self.frame_right, bg="white", bd=2, relief="raised")
        self.frame_instr.pack(fill="x", pady=(0, 15))
        tk.Label(self.frame_instr, text="🎮 Monitor en Tiempo Real", font=("Segoe UI", 12, "bold"), bg="white", fg="#0077b6").pack(pady=5)
        tk.Label(self.frame_instr, text="[ESPACIO] Pausa | [W/S] Velocidad | [ESC] Finalizar y Generar PDF", font=("Consolas", 10), bg="#e0fbfc", pady=10).pack(fill="x")

        # Consola
        tk.Label(self.frame_right, text="📟 Logs del Sistema", bg=COLOR_GLASS, font=("Consolas", 11, "bold")).pack(anchor="w")
        self.console = scrolledtext.ScrolledText(self.frame_right, height=20, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 10))
        self.console.pack(fill="both", expand=True)

    # --- LÓGICA ---
    def analisis_completo(self):
        ruta = filedialog.askopenfilename(title="Seleccionar Video", filetypes=[("MP4", "*.mp4"), ("Todos", "*.*")])
        if not ruta: return

        self.console.insert(tk.END, f"\n[!] Video seleccionado: {os.path.basename(ruta)}\n")
        
        # COMANDO MAESTRO: Llama al Smart Player con la bandera --report activada
        # Esto hace que muestre el video Y genere el PDF al final.
        cmd = [
            sys.executable, "Smart_Player.py", 
            "--source", ruta, 
            "--model", "yolov8s.pt",  # Usamos el modelo bueno para el reporte
            "--report"                # ACTIVAMOS EL REPORTE PDF
        ]
        
        desc = "MODO MAESTRO:\n1. Reproducción en vivo.\n2. Análisis estadístico en tiempo real.\n3. Generación automática de PDF al cerrar."
        self.iniciar_experimento(cmd, desc)

    def crear_boton(self, texto, comando, desc):
        btn = tk.Button(self.frame_left, text=texto, bg="#e1f5fe", fg="#01579b", relief="flat", cursor="hand2",
                        command=lambda: self.iniciar_experimento(comando, desc))
        btn.pack(pady=5, padx=20, fill="x")
        btn.bind("<Enter>", lambda e: self.actualizar_info(desc))

    def actualizar_info(self, texto):
        self.lbl_info_texto.config(text=texto)

    def iniciar_experimento(self, comando, desc):
        self.actualizar_info(desc)
        if self.current_process:
            messagebox.showwarning("Ocupado", "Detén el proceso actual primero.")
            return
        
        self.console.delete(1.0, tk.END)
        self.console.insert(tk.END, "> Iniciando sistema...\n")
        self.btn_stop.config(state="normal", bg="#ff5252", fg="white")
        t = threading.Thread(target=self.correr_script, args=(comando,))
        t.start()

    def correr_script(self, comando):
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        self.current_process = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                                text=True, encoding='utf-8', errors='replace', creationflags=flags)
        while True:
            try:
                line = self.current_process.stdout.readline()
                if not line and self.current_process.poll() is not None: break
                if line: self.console.insert(tk.END, line); self.console.see(tk.END)
            except: continue
        self.console.insert(tk.END, "\n> FINALIZADO.\n")
        self.limpiar_estado()

    def detener_proceso(self):
        if self.current_process:
            self.current_process.kill()
            self.current_process = None
            self.console.insert(tk.END, "\n⛔ DETENIDO.\n")
            self.limpiar_estado()

    def limpiar_estado(self):
        self.current_process = None
        self.btn_stop.config(state="disabled", bg="#ffcccc", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = AeroDashboard(root)
    root.mainloop()