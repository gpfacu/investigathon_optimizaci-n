"""
inverso_s7.py — el "camino inverso": fijar μ y buscar el estado más estable con esa densidad.

------------------------------------------------------------------------------------------
La idea
------------------------------------------------------------------------------------------
Hasta ahora las búsquedas fijaban un margen de estabilidad (eps) y maximizaban μ.
Acá es al revés: se fija una densidad objetivo μ* (por ejemplo 0.6935, por encima del
récord 0.6929077) y se busca la plantilla con ESE grado mínimo que tenga el mayor margen
de estabilidad λ2 posible.

  - margen > 0  -> existe un estado no sincronizado ESTABLE con grado mínimo ≥ μ*:
                   un grafo con esa densidad que NO sincroniza globalmente.
                   Si μ* > 0.6929077, eso mejoraría la cota.
  - margen < 0  -> con lo que se probó, ninguna plantilla de esa densidad es estable
                   (evidencia, no prueba: la búsqueda es local).

Barriendo muchos μ* se obtiene una CURVA margen(μ*): baja a medida que μ* sube, y el punto
donde cruza el cero es la máxima densidad alcanzable con estas plantillas.

------------------------------------------------------------------------------------------
Cómo se calcula (para cada configuración de grupos x, φ)
------------------------------------------------------------------------------------------
max_margen(x, φ, μ*) resuelve, con el mismo LP y planos de corte que plantillas.py:
    maximizar  s
    sujeto a   x_a + Σ_b P_ab x_b ≥ μ*            (grado mínimo ≥ μ*)
               Σ_b P_ab x_b sin(φ_a − φ_b) = 0    (equilibrio)
               x_a + Σ_b P_ab x_b cos(φ_a − φ_b) ≥ 1e-6   (nadie se escapa de su grupo)
               L(y) ≥ s  para cada vector y de la lista de cortes   (modos, ver abajo)
               0 ≤ P ≤ 1
Planos de corte: se resuelve el LP, se calcula el λ2 de verdad de la P obtenida
(plantillas.modos_normalizados); si es menor que el s que prometió el LP, se agrega el
corte L(y) ≥ s con su autovector y se repite. Devuelve:
    margen = λ2 real de la mejor P encontrada (una cota INFERIOR del óptimo: es alcanzable)
    cota   = el s del último LP (una cota SUPERIOR: no se puede más que eso)
Si el LP no tiene solución (con esos x, φ no se llega al grado μ* en equilibrio), None.

La búsqueda (buscar_margen) es el mismo recocido simulado de plantillas.buscar, con las
mismas jugadas (plantillas.proponer), pero el puntaje es el margen. Como el margen va de
~1e-2 a ~1e-12, la temperatura es RELATIVA: empeorar un 5% del margen actual se acepta con
probabilidad e^-1. Arranca de miembros al azar de la familia S7 (plantilla_s7.s7_aleatoria)
o de la mejor plantilla encontrada para ese μ* o para el μ* inmediatamente inferior.

Al final, los márgenes positivos de cada μ* se VERIFICAN en alta precisión (60 dígitos,
plantilla_s7.mejor_densidad_alta): se confirma que con ese margen hay una plantilla de
grado ≥ μ*. Márgenes por debajo de ~1e-10 solo son confiables después de esa verificación.

------------------------------------------------------------------------------------------
Uso
------------------------------------------------------------------------------------------
  python inverso_s7.py --minutos 300 --carpeta inverso
  python inverso_s7.py resumen --carpeta inverso        (tabla, gráfico y verificación)
Necesita plantillas.py y plantilla_s7.py en la misma carpeta (no los modifica).
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from datetime import datetime

import numpy as np
from scipy.optimize import linprog

import plantillas as T
import plantilla_s7 as S7

MU0 = S7.MU0
OBJETIVOS = [0.680, 0.685, 0.688, 0.690, 0.691, 0.692, 0.6925, 0.6928, 0.6929,
             0.69291, 0.6930, 0.6935, 0.6950, 0.7000]
COLUMNAS = ["id", "fecha", "mu_obj", "margen", "cota", "m_final", "inicio", "semilla",
            "iters", "temp", "segundos", "evaluadas", "podadas"]


# ======================================================================================
# 1. El LP inverso: máximo margen con grado ≥ μ*
# ======================================================================================
def max_margen(x, phi, mu_obj, max_cortes=300, umbral=-np.inf, c_min=1e-6):
    """Ver el encabezado. Devuelve (margen, cota, P) o (None, None, None)."""
    x, phi = np.asarray(x, float), np.asarray(phi, float)
    m = len(x)
    E, G, Fs, Cc, w = T._matrices(x, phi)
    k = len(E)                                            # variables: P (k) y s
    A_ub = [np.hstack([-G, np.zeros((m, 1))]),            # grado ≥ μ*
            np.hstack([-Cc, np.zeros((m, 1))])]           # c_a ≥ c_min
    b_ub = [x - mu_obj, x - c_min]
    A_eq = np.hstack([Fs, np.zeros((m, 1))])
    bounds = [(0, 1)] * k + [(-1, 1)]
    obj = np.zeros(k + 1); obj[-1] = -1                   # maximizar s
    cortes, mejor = [], (None, None, None)
    for _ in range(max_cortes):
        r = linprog(obj, A_ub=np.vstack(A_ub + cortes), b_ub=np.concatenate(b_ub + [np.zeros(len(cortes))]),
                    A_eq=A_eq, b_eq=np.zeros(m), bounds=bounds, method="highs", options=T.OPCIONES_LP)
        if r.status != 0:
            return mejor if mejor[0] is not None else (None, None, None)
        s_lp = r.x[-1]
        if s_lp < umbral:                                 # poda: ni la cota llega
            return (None, None, None)
        P = T._armar_P(r.x[:k], E, m)
        lam, Y = T.modos_normalizados(x, phi, P)
        if mejor[0] is None or lam[0] > mejor[0]:
            mejor = (lam[0], s_lp, P)
        else:
            mejor = (mejor[0], s_lp, mejor[2])
        if lam[0] >= s_lp - max(1e-13, 1e-4 * abs(s_lp)):  # el λ2 real alcanza lo prometido
            return mejor
        for i in range(m - 1):
            if lam[i] < s_lp:
                cortes.append(np.append(-T._fila_corte(Y[:, i], E, w), 1.0)[None, :])
    return mejor


# ======================================================================================
# 2. La búsqueda: recocido simulado sobre (x, φ) maximizando el margen
# ======================================================================================
def buscar_margen(x0, phi0, mu_obj, iters=1500, temp=0.05, m_max=9, semilla=0,
                  escala_min=-8, estadisticas=None):
    """
    Recocido simulado con temperatura relativa: una jugada que baja el margen en Δ se acepta
    con probabilidad exp(−Δ / (temp·escala)), con escala = max(|margen actual|, 1e-12), y
    nunca si Δ > 25·temp·escala (eso permite la poda). Devuelve (margen, cota, x, φ, P).
    """
    rng = np.random.default_rng(semilla)
    x, phi = T.normalizar(np.asarray(x0, float), np.asarray(phi0, float))
    s, cota, P = max_margen(x, phi, mu_obj)
    if s is None:
        s, cota, P = -np.inf, None, None
    mejor = (s, cota, x, phi, P)
    est = {"evaluadas": 0, "podadas": 0}
    for _ in range(iters):
        x2, phi2, _ = T.proponer(x, phi, rng, m_max, escala_min)
        if x2 is None:
            continue
        esc = temp * max(abs(s), 1e-12) if np.isfinite(s) else np.inf
        umbral = s - 25 * esc if np.isfinite(s) else -np.inf
        s2, c2, P2 = max_margen(x2, phi2, mu_obj, umbral=umbral)
        est["evaluadas"] += 1
        if s2 is None:
            est["podadas"] += 1
            continue
        if s2 >= s or (np.isfinite(s) and s2 >= umbral and rng.random() < np.exp((s2 - s) / esc)):
            x, phi, s = x2, phi2, s2
            if s > mejor[0]:
                mejor = (s, c2, x, phi, P2)
    if estadisticas is not None:
        estadisticas.update(est)
    return mejor


# ======================================================================================
# 3. El barrido: muchas búsquedas en paralelo, repartidas entre los μ* objetivo
# ======================================================================================
def leer(carpeta):
    ruta = os.path.join(carpeta, "resultados.csv")
    if not os.path.exists(ruta):
        return []
    with open(ruta, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def mejor_por_objetivo(filas):
    """{μ*: fila con mayor margen} para las búsquedas que encontraron algo."""
    mejores = {}
    for r in filas:
        if r["margen"] in ("", "None"):
            continue
        mo, mg = float(r["mu_obj"]), float(r["margen"])
        if mo not in mejores or mg > float(mejores[mo]["margen"]):
            mejores[mo] = r
    return mejores


def elegir_inicio(rng, carpeta, mu_obj):
    """
    50%: arranca de la mejor plantilla ya encontrada para este μ* o para el μ* de abajo
         (una plantilla buena para 0.6925 es un buen punto de partida para 0.6928);
    50%: un miembro al azar de la familia S7 del paper.
    """
    mejores = mejor_por_objetivo(leer(carpeta))
    previos = [mo for mo in mejores if mo <= mu_obj and os.path.exists(os.path.join(carpeta, mejores[mo]["id"] + ".npz"))]
    if previos and rng.random() < 0.5:
        mo = max(previos) if rng.random() < 0.5 or mu_obj not in previos else mu_obj
        d = np.load(os.path.join(carpeta, mejores[mo]["id"] + ".npz"))
        return f"continúa {mejores[mo]['id']} (μ*={mo})", d["x"], d["phi"]
    return S7.s7_aleatoria(rng)


def una_busqueda(t):
    t0 = time.time()
    est = {}
    s, cota, x, phi, P = buscar_margen(t["x0"], t["phi0"], t["mu_obj"], iters=t["iters"], temp=t["temp"],
                                       m_max=t["m_max"], semilla=t["semilla"], escala_min=t["escala_min"],
                                       estadisticas=est)
    ok = np.isfinite(s)
    fila = {"id": t["id"], "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "mu_obj": t["mu_obj"],
            "margen": s if ok else "", "cota": cota if ok else "", "m_final": len(x), "inicio": t["inicio"],
            "semilla": t["semilla"], "iters": t["iters"], "temp": t["temp"],
            "segundos": round(time.time() - t0, 1), "evaluadas": est["evaluadas"], "podadas": est["podadas"]}
    np.savez(os.path.join(t["carpeta"], t["id"] + ".npz"), x=x, phi=phi,
             P=P if P is not None else np.zeros((len(x), len(x))), mu_obj=t["mu_obj"],
             margen=s if ok else np.nan, x0=t["x0"], phi0=t["phi0"], semilla=t["semilla"],
             iters=t["iters"], temp=t["temp"], m_max=t["m_max"], escala_min=t["escala_min"])
    return fila


def guardar(carpeta, fila):
    ruta = os.path.join(carpeta, "resultados.csv")
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def escribir_resumen(carpeta, verificar=False):
    """resumen.txt: el mejor margen para cada μ*. Con verificar=True agrega la verificación
    en alta precisión de los márgenes positivos."""
    mejores = mejor_por_objetivo(leer(carpeta))
    lineas = [f"Camino inverso: mayor margen de estabilidad encontrado para cada densidad μ*",
              f"Récord del paper: μ0 = {MU0:.10f}.  margen > 0 = existe estado trabado estable "
              f"(el grafo NO sincroniza)", ""]
    lineas.append(f"{'μ*':>9}  {'margen (λ2)':>13}  {'cota sup.':>11}  {'grupos':>6}  {'estado':<34} {'id'}")
    for mo in sorted(mejores):
        r = mejores[mo]
        mg = float(r["margen"])
        if mg > 1e-10:
            estado = "estable: NO sincroniza"
        elif mg > 0:
            estado = "estable, bajo el piso numérico"
        else:
            estado = "no se encontró estado estable"
        if mo > MU0:
            estado += "  (> récord)"
        ver = ""
        if verificar and mg > 0:
            d = np.load(os.path.join(carpeta, r["id"] + ".npz"))
            x, phi = S7._ordenar_por_fase(d["x"], d["phi"])
            z = S7.mejor_densidad_alta(x, phi, mg * 0.9, 60)
            if z is None:
                ver = "   alta precisión: NO confirmado"
            else:
                mu_alta = float(z["mu"])
                ver = (f"   alta precisión: μ = {mu_alta:.10f} con λ2 = {float(z['lambda2']):.1e} "
                       + ("-> CONFIRMADO" if mu_alta >= mo - 1e-12 else "-> no alcanza μ*"))
        lineas.append(f"{mo:9.5f}  {mg:13.4e}  {float(r['cota']):11.3e}  {r['m_final']:>6}  {estado:<34} {r['id']}{ver}")
    texto = "\n".join(lineas) + "\n"
    with open(os.path.join(carpeta, "resumen.txt"), "w", encoding="utf-8") as f:
        f.write(texto)
    return texto, mejores


def graficar(carpeta, mejores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mo = np.array(sorted(mejores))
    mg = np.array([float(mejores[v]["margen"]) for v in mo])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    pos = mg > 0
    axs[0].semilogy(mo[pos], mg[pos], "o-", color="#2a78d6")
    axs[0].set_title("Margen de estabilidad (solo los positivos, escala log)", fontsize=10, loc="left")
    axs[1].plot(mo, np.sign(mg) * np.log10(1 + np.abs(mg) / 1e-12), "o-", color="#2a78d6")
    axs[1].axhline(0, color="#52514e", lw=1)
    axs[1].set_title("Signo del margen (>0: no sincroniza)  ·  escala log simétrica", fontsize=10, loc="left")
    axs[0].set_ylabel("margen λ2")
    axs[1].set_ylabel("signo · log10(1 + |margen| / 1e-12)")
    for ax in axs:
        ax.axvline(MU0, color="#0b0b0b", ls="--", lw=1)
        ax.text(MU0, ax.get_ylim()[1], " récord", fontsize=8, va="top")
        ax.set_xlabel("densidad objetivo μ*")
        ax.grid(color="#e4e3df", lw=0.8)
    plt.tight_layout()
    ruta = os.path.join(carpeta, "curva_margen.png")
    plt.savefig(ruta, dpi=110)
    return ruta


def main():
    ap = argparse.ArgumentParser(description="Camino inverso: máximo margen para cada μ*")
    ap.add_argument("--minutos", type=float, default=60)
    ap.add_argument("--procesos", type=int, default=os.cpu_count())
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--m_max", type=int, default=9)
    ap.add_argument("--escala_min", type=float, default=-8)
    ap.add_argument("--objetivos", default=",".join(str(v) for v in OBJETIVOS),
                    help="lista de μ* separados por coma")
    ap.add_argument("--carpeta", default="inverso")
    a = ap.parse_args()
    objetivos = [float(v) for v in a.objetivos.split(",")]
    os.makedirs(a.carpeta, exist_ok=True)
    rng = np.random.default_rng()
    fin = time.time() + 60 * a.minutos
    contador = 0
    print(f"Camino inverso: {len(objetivos)} densidades objetivo, {a.minutos:g} min, {a.procesos} procesos, "
          f"carpeta '{a.carpeta}/'. Ctrl+C corta sin perder nada.")
    print(f"Récord del paper: μ0 = {MU0:.10f}\n")

    def tarea():
        nonlocal contador
        mo = objetivos[contador % len(objetivos)]          # se reparten por turnos
        contador += 1
        inicio, x0, phi0 = elegir_inicio(rng, a.carpeta, mo)
        return {"id": datetime.now().strftime("%m%d_%H%M%S") + f"_{contador:04d}", "mu_obj": mo,
                "inicio": inicio, "x0": np.asarray(x0, float), "phi0": np.asarray(phi0, float),
                "semilla": int(rng.integers(1_000_000)), "iters": a.iters, "m_max": a.m_max,
                "escala_min": a.escala_min, "temp": float(rng.choice([0.02, 0.05, 0.1])), "carpeta": a.carpeta}

    try:
        with ProcessPoolExecutor(a.procesos) as pool:
            pend = {pool.submit(una_busqueda, tarea()) for _ in range(a.procesos)}
            while pend:
                listos, pend = wait(pend, return_when=FIRST_COMPLETED)
                for f in listos:
                    fila = f.result()
                    guardar(a.carpeta, fila)
                    mg = fila["margen"]
                    txt = "sin solución" if mg == "" else f"margen = {mg:+.3e}"
                    marca = "  <- estable" if mg != "" and mg > 0 else ""
                    if mg != "" and mg > 0 and fila["mu_obj"] > MU0:
                        marca += "  ‼ POR ENCIMA DEL RÉCORD (verificar con 'resumen')"
                    print(f"[{fila['id']}] μ* = {fila['mu_obj']:.5f}  {txt}  ({fila['m_final']} grupos, "
                          f"{fila['segundos']:.0f} s){marca}", flush=True)
                    escribir_resumen(a.carpeta)
                    if time.time() < fin:
                        pend.add(pool.submit(una_busqueda, tarea()))
    except KeyboardInterrupt:
        print("\nCortado. Lo terminado ya está guardado.")
    texto, mejores = escribir_resumen(a.carpeta)
    print("\n" + texto)
    print("Para la tabla con verificación en alta precisión y el gráfico:")
    print(f"  python inverso_s7.py resumen --carpeta {a.carpeta}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "resumen":
        ap = argparse.ArgumentParser(prog="inverso_s7.py resumen")
        ap.add_argument("--carpeta", default="inverso")
        a = ap.parse_args(sys.argv[2:])
        texto, mejores = escribir_resumen(a.carpeta, verificar=True)
        print(texto)
        if mejores:
            print("Gráfico:", graficar(a.carpeta, mejores))
    else:
        main()
