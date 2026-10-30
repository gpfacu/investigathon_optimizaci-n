"""
correr_noche.py — deja todo corriendo solo, en orden, y guarda cada salida en un archivo.

  1) Búsqueda desde la familia S7       (plantilla_s7.py)        -> carpeta noche_s7/
  2) Refinamiento en alta precisión      (plantilla_s7.py refinar) de las 5 mejores
  3) Camino inverso, barrido de μ*       (inverso_s7.py)          -> carpeta noche_inverso/
  4) Resumen del camino inverso, con verificación en alta precisión y gráfico

Uso (10 horas en total, por defecto 5 h + 4.5 h + unos minutos de refinamiento y resumen):
    python correr_noche.py
    python correr_noche.py --horas_s7 5 --horas_inverso 4.5 --procesos 7

Todo lo que se ve en pantalla queda también en noche_log.txt. Si algo se corta, lo ya
guardado no se pierde; volver a correr retoma desde los mejores resultados.
Necesita en la misma carpeta: plantillas.py, busqueda.py, plantilla_s7.py, inverso_s7.py.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("--horas_s7", type=float, default=5.0)
ap.add_argument("--horas_inverso", type=float, default=4.5)
ap.add_argument("--procesos", type=int, default=os.cpu_count())
a = ap.parse_args()

# En Windows, al mandar la salida a un archivo Python usa la codificación del sistema y los
# caracteres como μ o ★ pueden romper el programa. Se fuerza UTF-8 en los procesos hijos.
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
log = open("noche_log.txt", "a", encoding="utf-8")


def escribir(texto):
    log.write(texto); log.flush()
    sys.stdout.write(texto.encode(sys.stdout.encoding or "utf-8", errors="replace")
                     .decode(sys.stdout.encoding or "utf-8", errors="replace"))
    sys.stdout.flush()


def paso(titulo, args):
    escribir(f"\n{'=' * 90}\n{datetime.now():%Y-%m-%d %H:%M}  {titulo}\n  python {' '.join(args)}\n{'=' * 90}\n")
    t0 = time.time()
    # "-u": salida sin búfer, para que cada línea aparezca en el log apenas se imprime
    p = subprocess.Popen([sys.executable, "-u"] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env=env, text=True, encoding="utf-8", errors="replace")
    for linea in p.stdout:
        escribir(linea)
    p.wait()
    escribir(f"-> terminó en {(time.time() - t0) / 60:.1f} min (código {p.returncode})\n")


paso("1) Búsqueda desde la familia S7",
     ["plantilla_s7.py", "--minutos", str(60 * a.horas_s7), "--iters", "3000", "--m_max", "9",
      "--eps", "1e-9", "--escala_min", "-8", "--procesos", str(a.procesos), "--carpeta", "noche_s7"])
paso("2) Refinamiento en alta precisión de las 5 mejores",
     ["plantilla_s7.py", "refinar", "--carpeta", "noche_s7", "--top", "5"])
paso("3) Camino inverso: máximo margen para cada μ*",
     ["inverso_s7.py", "--minutos", str(60 * a.horas_inverso), "--iters", "1500", "--m_max", "9",
      "--escala_min", "-8", "--procesos", str(a.procesos), "--carpeta", "noche_inverso"])
paso("4) Resumen del camino inverso (verificación en alta precisión + gráfico)",
     ["inverso_s7.py", "resumen", "--carpeta", "noche_inverso"])
escribir(f"\nListo: {datetime.now():%Y-%m-%d %H:%M}. Resultados en noche_s7/, noche_inverso/ y noche_log.txt\n")
