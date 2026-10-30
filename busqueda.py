"""
Búsqueda en serio de plantillas con μ alto.  Uso:

    python busqueda.py                    # 30 minutos, usando todos los núcleos
    python busqueda.py --minutos 120      # más tiempo
    python busqueda.py --minutos 5 --iters 500   # prueba rápida

Qué hace:
  - Lanza muchas búsquedas cortas (`plantillas.buscar`) en paralelo, cada una desde un
    punto de partida distinto (S7, C4 con grupos partidos, Canale, twisted, o uno de los
    MEJORES resultados anteriores de cada familia) y con parámetros al azar.
  - Cada búsqueda usa el puntaje suave (puede atravesar configuraciones inestables) y
    la poda (descarta sin terminar de calcular las jugadas que seguro no se aceptan).
  - Cada búsqueda terminada se guarda enseguida en la carpeta `resultados/`:
        resultados/resultados.csv   una fila por búsqueda (para comparar)
        resultados/<id>.npz         la plantilla completa (x, φ, P, historia)
        resultados/resultados.txt   lo mismo que el csv pero para leer a ojo
        resultados/mejor.txt        la mejor plantilla hasta ahora, con todos sus números
    Así podés cortar con Ctrl+C en cualquier momento sin perder nada, y volver a
    lanzarlo después: las búsquedas nuevas arrancan desde los mejores resultados viejos.
  - Los resultados se miran con el notebook `ver_resultados.ipynb`.
"""
import argparse
import csv
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from datetime import datetime

import numpy as np
import plantillas as T

RECORD = 0.6929077       # cota de O'Keeffe (oct. 2026): lo que hay que superar
CANALE = 11 / 16

COLUMNAS = ["id", "fecha", "inicio", "familia", "semilla", "eps", "iters", "temp", "m_max",
            "m_final", "mu", "lambda2", "max_F", "segundos", "evaluadas", "podadas"]


# ---------------------------------------------------------------------------
# Puntos de partida
# ---------------------------------------------------------------------------
def leer_csv(carpeta):
    ruta = os.path.join(carpeta, "resultados.csv")
    if not os.path.exists(ruta):
        return []
    # errors="replace": en Windows las versiones anteriores escribían el csv en cp1252
    with open(ruta, newline="", encoding="utf-8", errors="replace") as f:
        return [r for r in csv.DictReader(f)]


FAMILIAS = ["s7", "c4_partido", "canale", "twisted"]
PESOS = [0.35, 0.35, 0.15, 0.15]      # cuánto se usa cada punto de partida nuevo


def agotados(filas, veces=2, tol=1e-8):
    """
    Ids de resultados "agotados": ya se continuaron `veces` veces o más y ninguna de esas
    continuaciones mejoró en más de `tol` (1e-8). Volver a continuarlos repite el mismo número
    (en la corrida de 5 h, un 25% del tiempo se fue en eso).
    Los resultados con el mismo μ (hasta 1e-8) en la misma familia se tratan como una
    sola plantilla: son copias que salieron de continuar el mismo punto.
    """
    mu = {r["id"]: float(r["mu"]) for r in filas}
    clase = {r["id"]: (r.get("familia"), round(float(r["mu"]), 8)) for r in filas}
    intentos = {}
    for r in filas:
        m = re.search(r"(\d{4}_\d{6}_\d+)", r["inicio"])
        if r["inicio"].startswith("contin") and m and m.group(1) in mu:
            padre = m.group(1)
            mejoro = float(r["mu"]) > mu[padre] + tol
            c = clase[padre]
            n, alguna = intentos.get(c, (0, False))
            intentos[c] = (n + 1, alguna or mejoro)
    clases_agotadas = {c for c, (n, alguna) in intentos.items() if n >= veces and not alguna}
    return {i for i, c in clase.items() if c in clases_agotadas}


def elegir_inicio(rng, carpeta):
    """
    Elige desde dónde arrancar. Devuelve (descripción, familia, x, φ).
    - 40% de las veces sigue desde un resultado anterior. Para no gastar todo el tiempo
      afinando la misma familia (lo que pasó en la primera corrida: todo era S7), primero
      elige una familia al azar entre las que tienen resultados, y después una de las 3
      mejores plantillas DISTINTAS de esa familia, salteando las agotadas (ver agotados()).
    - Si no, arranca desde una plantilla conocida: S7 con distintas separaciones h,
      C4 con 2-4 grupos partidos al azar, Canale, o twisted.
    """
    filas = [r for r in leer_csv(carpeta) if float(r["mu"]) > 0 and r.get("familia")]
    fuera = agotados(filas)
    candidatas = [r for r in filas if r["id"] not in fuera
                  and os.path.exists(os.path.join(carpeta, r["id"] + ".npz"))]
    if candidatas and rng.random() < 0.4:
        familia = rng.choice(sorted({r["familia"] for r in candidatas}))
        de_esa = sorted([r for r in candidatas if r["familia"] == familia], key=lambda r: -float(r["mu"]))
        # una sola copia por valor de μ, así los "3 mejores" son 3 plantillas distintas
        distintas, vistos = [], set()
        for r in de_esa:
            if round(float(r["mu"]), 8) not in vistos:
                vistos.add(round(float(r["mu"]), 8)); distintas.append(r)
        r = distintas[rng.integers(min(3, len(distintas)))]
        d = np.load(os.path.join(carpeta, r["id"] + ".npz"))
        return f"continúa {r['id']}", familia, d["x"], d["phi"]
    familia = str(rng.choice(FAMILIAS, p=PESOS))
    if familia == "canale":
        return "canale", familia, *T.canale()
    if familia == "s7":
        h = float(rng.choice([0.002, 0.005, 0.01, 0.02, 0.05]))
        return f"s7(h={h})", familia, *T.s7(h)
    if familia == "c4_partido":
        x, phi = T.c4_partido(rng)
        return f"c4_partido({len(x) - 4} partidos)", familia, x, phi
    m = int(rng.integers(8, 13))
    return f"twisted({m})", familia, *T.twisted(m)


# ---------------------------------------------------------------------------
# Una búsqueda (corre en un proceso aparte)
# ---------------------------------------------------------------------------
def una_busqueda(tarea):
    t0 = time.time()
    historia, est = [], {}
    mu, x, phi = T.buscar(tarea["x0"], tarea["phi0"], eps=tarea["eps"], iters=tarea["iters"],
                          m_max=tarea["m_max"], semilla=tarea["semilla"], temp=tarea["temp"],
                          log_cada=0, historia=historia, estadisticas=est,
                          escala_min=tarea["escala_min"])
    fila = {k: tarea[k] for k in ["id", "inicio", "familia", "semilla", "eps", "iters", "temp", "m_max"]}
    fila.update(fecha=datetime.now().strftime("%Y-%m-%d %H:%M"), m_final=len(x),
                mu=mu, lambda2="", max_F="", segundos=round(time.time() - t0, 1),
                evaluadas=est["evaluadas"], podadas=est["podadas"])
    P = None
    if mu > 0:
        _, P = T.mejor_densidad(x, phi, tarea["eps"])
        v = T.verificar(x, phi, P)
        fila.update(lambda2=v["lambda2"], max_F=v["max|F|"])
    np.savez(os.path.join(tarea["carpeta"], tarea["id"] + ".npz"), x=x, phi=phi,
             P=P if P is not None else np.zeros((len(x), len(x))), mu=mu,
             historia=np.array(historia), inicio=tarea["inicio"],
             # condiciones iniciales exactas de esta búsqueda, para poder repetirla
             x0=np.asarray(tarea["x0"], float), phi0=np.asarray(tarea["phi0"], float),
             semilla=tarea["semilla"], temp=tarea["temp"], eps=tarea["eps"],
             iters=tarea["iters"], m_max=tarea["m_max"], escala_min=tarea["escala_min"])
    return fila


def guardar_txt(carpeta, fila, marca):
    """resultados.txt: una línea legible por búsqueda (se va agregando, nunca se borra)."""
    ruta = os.path.join(carpeta, "resultados.txt")
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", encoding="utf-8") as f:
        if nuevo:
            f.write("Búsquedas de plantillas. Referencias: Canale 11/16 = 0.6875000, "
                    f"récord O'Keeffe = {RECORD:.7f}\n")
            f.write("fecha             id                  μ          grupos  segundos  inicio\n")
        mu_txt = "sin plantilla" if fila["mu"] < 0 else f"{fila['mu']:.7f}"
        f.write(f"{fila['fecha']}  {fila['id']}  {mu_txt:>10s}  {fila['m_final']:>6d}  "
                f"{fila['segundos']:>8.0f}  {fila['inicio']}{marca}\n")


def guardar_mejor_txt(carpeta):
    """mejor.txt: la mejor plantilla hasta ahora, completa (se reescribe cada vez que mejora)."""
    filas = [r for r in leer_csv(carpeta) if float(r["mu"]) > 0]
    if not filas:
        return
    r = max(filas, key=lambda r: float(r["mu"]))
    d = np.load(os.path.join(carpeta, r["id"] + ".npz"))
    orden = np.argsort(d["phi"])
    x, phi, P = d["x"][orden], d["phi"][orden], d["P"][np.ix_(orden, orden)]
    v = T.verificar(x, phi, P)
    with open(os.path.join(carpeta, "mejor.txt"), "w", encoding="utf-8") as f:
        f.write(f"Mejor plantilla hasta ahora: {r['id']}  (arrancó desde {r['inicio']})\n")
        f.write(f"μ = {v['mu']:.10f}   ({'SUPERA' if v['mu'] > RECORD else 'no supera'} "
                f"el récord {RECORD})\n")
        f.write(f"λ2 = {v['lambda2']:.3e}   max|F| = {v['max|F|']:.1e}   min c = {v['min c']:.4f}\n")
        f.write(f"(chequeo en punto flotante, NO es una prueba)\n\n")
        f.write("grupo   tamaño x_a      fase (rad)          fase (grados)     grado d_a\n")
        for a in range(len(x)):
            f.write(f"{a:5d}   {x[a]:.10f}   {phi[a]:.15f}   {np.degrees(phi[a]):14.10f}   "
                    f"{(P @ x)[a]:.10f}\n")
        f.write("\nDensidades P[a, b] (fracción de aristas entre los grupos a y b):\n")
        for a in range(len(x)):
            f.write("  " + "  ".join(f"{p:.6f}" for p in P[a]) + "\n")


def guardar_fila(carpeta, fila):
    ruta = os.path.join(carpeta, "resultados.csv")
    if os.path.exists(ruta):
        with open(ruta, newline="", encoding="utf-8", errors="replace") as f:
            encabezado = next(csv.reader(f), [])
        if encabezado != COLUMNAS:
            # csv de una versión anterior del programa: se guarda aparte y se empieza otro
            os.replace(ruta, os.path.join(carpeta, "resultados_version_vieja.csv"))
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        if nuevo:
            w.writeheader()
        w.writerow(fila)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Búsqueda de plantillas con μ alto")
    ap.add_argument("--minutos", type=float, default=30, help="tiempo total (default 30)")
    ap.add_argument("--procesos", type=int, default=os.cpu_count(), help="búsquedas en paralelo")
    ap.add_argument("--iters", type=int, default=2000, help="pasos por búsqueda (default 2000)")
    ap.add_argument("--eps", type=float, default=1e-8, help="margen de estabilidad λ2 (default 1e-8)")
    ap.add_argument("--m_max", type=int, default=12, help="máximo de grupos (default 12)")
    ap.add_argument("--escala_min", type=float, default=-8,
                    help="exponente de la jugada más chica: 10^escala_min (default -8)")
    ap.add_argument("--carpeta", default="resultados")
    args = ap.parse_args()

    os.makedirs(args.carpeta, exist_ok=True)
    rng = np.random.default_rng()
    fin = time.time() + 60 * args.minutos
    previos = [float(r["mu"]) for r in leer_csv(args.carpeta)]
    mejor = max(previos, default=-1)
    contador = 0

    print(f"Búsqueda de {args.minutos:g} min con {args.procesos} procesos. "
          f"Resultados en '{args.carpeta}/'. Ctrl+C para cortar (no se pierde nada).")
    print(f"Referencias: Canale = {CANALE:.7f}   récord O'Keeffe = {RECORD:.7f}")
    if previos:
        print(f"Hay {len(previos)} búsquedas anteriores; mejor μ hasta ahora = {mejor:.7f}\n")

    def nueva_tarea():
        nonlocal contador
        contador += 1
        inicio, familia, x0, phi0 = elegir_inicio(rng, args.carpeta)
        return dict(id=datetime.now().strftime("%m%d_%H%M%S") + f"_{contador:03d}",
                    inicio=inicio, familia=familia, x0=x0, phi0=phi0, eps=args.eps, iters=args.iters,
                    m_max=args.m_max, semilla=int(rng.integers(1_000_000)), escala_min=args.escala_min,
                    temp=float(rng.choice([1e-5, 1e-6, 1e-7])), carpeta=args.carpeta)

    try:
        with ProcessPoolExecutor(args.procesos) as pool:
            pendientes = {pool.submit(una_busqueda, nueva_tarea()) for _ in range(args.procesos)}
            while pendientes:
                listos, pendientes = wait(pendientes, return_when=FIRST_COMPLETED)
                for fut in listos:
                    fila = fut.result()
                    guardar_fila(args.carpeta, fila)
                    marca = ""
                    if fila["mu"] > mejor:
                        mejor, marca = fila["mu"], "  ★ nuevo mejor"
                    if fila["mu"] > RECORD:
                        marca += "  ‼ SUPERA EL RÉCORD (verificar!)"
                    guardar_txt(args.carpeta, fila, marca)
                    if "★" in marca:
                        guardar_mejor_txt(args.carpeta)
                    mu_txt = "sin plantilla" if fila["mu"] < 0 else f"μ = {fila['mu']:.7f}"
                    print(f"[{fila['id']}] {fila['inicio']:22s} {mu_txt}  "
                          f"({fila['m_final']} grupos, {fila['segundos']:.0f} s, "
                          f"{fila['podadas']}/{fila['evaluadas']} podadas){marca}", flush=True)
                    if time.time() < fin:
                        pendientes.add(pool.submit(una_busqueda, nueva_tarea()))
    except KeyboardInterrupt:
        print("\nCortado. Las búsquedas terminadas ya están guardadas.")

    filas = sorted(leer_csv(args.carpeta), key=lambda r: -float(r["mu"]))
    print("\nTop 5 plantillas distintas (entre paréntesis, cuántas búsquedas llegaron a ese mismo valor):")
    valores = {}
    for r in filas:
        valores.setdefault(round(float(r["mu"]), 11), []).append(r)
    for v, rs in list(valores.items())[:5]:
        r = rs[0]
        print(f"  μ = {float(r['mu']):.9f}  ({len(rs):3d})  {r['m_final']:>2s} grupos   "
              f"{r['id']}   familia {r.get('familia', '?')}")
    print("\nPara mirarlos: abrí ver_resultados.ipynb")


if __name__ == "__main__":
    main()
