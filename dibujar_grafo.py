"""
dibujar_grafo.py — dibuja una plantilla guardada como un grafo de verdad: circulitos y rayas.

Cada .npz que guardan busqueda.py, plantilla_s7.py e inverso_s7.py (y por lo tanto
correr_noche.py, en noche_s7/ y noche_inverso/) tiene la plantilla completa:
    x  = tamaño de cada grupo,   phi = fase de cada grupo,   P = densidad de aristas entre grupos.
Con eso se puede armar un grafo concreto de n vértices y dibujarlo.

Uso:
    python dibujar_grafo.py noche_s7/1026_202743_003.npz
    python dibujar_grafo.py noche_s7/1026_202743_003.npz --n 60 --exagerar 40
    python dibujar_grafo.py noche_s7 --mejor        (dibuja la de mayor μ de la carpeta)
    python dibujar_grafo.py noche_s7 --mejor --video            (video de cómo evolucionó)
    python dibujar_grafo.py noche_s7 --mejor --video --cadena   (toda la cadena de búsquedas)
Desde un notebook:
    from dibujar_grafo import dibujar, video
    dibujar("noche_s7/1026_202743_003.npz", n=40)
    video("noche_s7/1026_202743_003.npz")

Genera una imagen con dos paneles:
  - Izquierda, la PLANTILLA: un círculo por grupo, en su fase sobre la circunferencia, con
    área proporcional a su tamaño, y una raya por par de grupos conectados, más gruesa cuanto
    más aristas hay (naranja = densidad intermedia, gris = completa).
  - Derecha, un GRAFO CONCRETO de n vértices armado con esa plantilla: cada grupo se vuelve
    un racimo de vértices (circulitos del color del grupo) y cada arista es una raya.

Cómo se arma el grafo (la idea del Lema 2 del paper, en chico):
    - el grupo a recibe n_a ≈ x_a · n vértices (redondeado);
    - dentro de un grupo, todos conectados con todos (P_aa = 1);
    - entre los grupos a y b, el vértice i de a se conecta con los vértices j de b tales que
      (j − i·n_b/n_a) mod n_b < P_ab · n_b. Es un patrón "en anillo": cada vértice de a tiene
      la misma cantidad de vecinos en b (≈ P_ab · n_b), como pide la plantilla.

OJO: es una ilustración. Con n chico, los tamaños y las densidades se redondean, y los estados
de tipo S7 son tan frágiles (márgenes de 1e-9) que ese grafo chico NO conserva el estado
trabado. El μ del grafo dibujado se informa en el título, junto al μ de la plantilla.
"""
import argparse
import csv
import glob
import os

import numpy as np


def armar_grafo(x, phi, P, n):
    """Devuelve (grupo de cada vértice, matriz de adyacencia n×n) siguiendo la plantilla."""
    x = np.asarray(x, float) / np.sum(x)
    m = len(x)
    tam = np.maximum(1, np.round(x * n).astype(int))
    grupo = np.repeat(np.arange(m), tam)
    N = len(grupo)
    inicio = np.r_[0, np.cumsum(tam)[:-1]]
    A = np.zeros((N, N), dtype=np.uint8)
    for a in range(m):
        ia = inicio[a] + np.arange(tam[a])
        A[np.ix_(ia, ia)] = 1                                     # completo por dentro
        for b in range(a + 1, m):
            k = int(round(P[a, b] * tam[b]))                      # vecinos en b de cada vértice de a
            if k == 0:
                continue
            for i in range(tam[a]):
                desde = int(round(i * tam[b] / tam[a]))
                js = inicio[b] + (desde + np.arange(k)) % tam[b]
                A[ia[i], js] = 1
                A[js, ia[i]] = 1
    np.fill_diagonal(A, 0)
    return grupo, A


def _posiciones_plantilla(phi, exagerar):
    """Fase de dibujo de cada grupo: el cuarto de vuelta más cercano + la diferencia exagerada."""
    cuarto = np.round(phi / (np.pi / 2)) * (np.pi / 2)
    return cuarto + exagerar * (phi - cuarto)


def dibujar(ruta, n=40, exagerar=40, guardar=None, mostrar=True):
    import matplotlib
    if not mostrar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(ruta, allow_pickle=True)
    x, phi, P = d["x"], np.mod(d["phi"], 2 * np.pi), d["P"]
    o = np.argsort(phi)
    x, phi, P = x[o] / np.sum(x), phi[o], P[np.ix_(o, o)]
    m = len(x)
    if not P.any():
        raise ValueError("Este .npz no tiene plantilla estable (P vacía): no hay nada que dibujar.")
    mu_plantilla = float(np.min(P @ x))
    colores = plt.cm.tab10(np.arange(m) % 10)
    ang = _posiciones_plantilla(phi, exagerar)
    cx, cy = np.cos(ang), np.sin(ang)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5))

    # --- izquierda: la plantilla ---
    ax1.add_patch(plt.Circle((0, 0), 1, fill=False, color="#cfcfcf"))
    for a in range(m):
        for b in range(a + 1, m):
            if P[a, b] > 1e-3:
                col = "#9a9a9a" if P[a, b] > 0.98 else "#eb6834"
                ax1.plot([cx[a], cx[b]], [cy[a], cy[b]], color=col, alpha=0.8,
                         lw=0.5 + 60 * x[a] * x[b] * P[a, b], zorder=1)
    ax1.scatter(cx, cy, s=4000 * x, color=colores, edgecolor="white", linewidth=1.5, zorder=2)
    for a in range(m):
        ax1.text(1.22 * cx[a], 1.22 * cy[a], f"{np.degrees(phi[a]):.3f}°\nx={x[a]:.3f}",
                 ha="center", va="center", fontsize=7.5, color="#52514e")
    ax1.set_title(f"Plantilla: {m} grupos, μ = {mu_plantilla:.7f}\n"
                  f"(separaciones dentro de cada cuarto exageradas ×{exagerar}; "
                  f"naranja = aristas raleadas)", fontsize=10)

    # --- derecha: un grafo concreto ---
    grupo, A = armar_grafo(x, phi, P, n)
    N = len(grupo)
    pos = np.zeros((N, 2))
    radio = 0.06 + 0.25 * np.sqrt(x)                          # racimo más grande para grupos grandes
    for a in range(m):
        idx = np.where(grupo == a)[0]
        t = 2 * np.pi * np.arange(len(idx)) / max(len(idx), 1)
        pos[idx, 0] = cx[a] + radio[a] * np.cos(t)
        pos[idx, 1] = cy[a] + radio[a] * np.sin(t)
    ii, jj = np.nonzero(np.triu(A, 1))
    mismo = grupo[ii] == grupo[jj]
    for (i, j, s) in zip(ii, jj, mismo):
        ax2.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                 color="#bdbdbd" if not s else colores[grupo[i]], alpha=0.12 if not s else 0.35,
                 lw=0.5, zorder=1)
    ax2.scatter(pos[:, 0], pos[:, 1], s=28, color=colores[grupo], edgecolor="white", linewidth=0.6, zorder=2)
    grado = A.sum(1)
    mu_grafo = grado.min() / (N - 1)
    ax2.set_title(f"Grafo concreto de {N} vértices y {len(ii)} aristas armado con la plantilla\n"
                  f"grado mínimo {grado.min()}  ->  μ del grafo = δ/(n−1) = {mu_grafo:.4f} "
                  f"(redondeo con n chico)", fontsize=10)

    for ax in (ax1, ax2):
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    fig.suptitle(nombre, fontsize=11)
    plt.tight_layout()
    if guardar is None:
        guardar = os.path.splitext(ruta)[0] + "_grafo.png"
    plt.savefig(guardar, dpi=120)
    if mostrar:
        plt.show()
    plt.close(fig)
    return guardar


def mejor_de_carpeta(carpeta):
    """La plantilla de mayor μ (o de mayor margen en una carpeta del camino inverso)."""
    ruta = os.path.join(carpeta, "resultados.csv")
    with open(ruta, newline="", encoding="utf-8", errors="replace") as f:
        filas = list(csv.DictReader(f))
    if filas and "margen" in filas[0]:
        filas = [r for r in filas if r["margen"] not in ("", "None") and float(r["margen"]) > 0]
        r = max(filas, key=lambda r: (float(r["mu_obj"]), float(r["margen"])))
    else:
        r = max([r for r in filas if float(r["mu"]) > 0], key=lambda r: float(r["mu"]))
    return os.path.join(carpeta, r["id"] + ".npz")


# ======================================================================================
# Video: re-correr la búsqueda y sacar una foto en cada cambio aceptado
# ======================================================================================
# La búsqueda es determinista: con el mismo punto de partida (x0, φ0), la misma semilla y los
# mismos parámetros hace exactamente los mismos pasos. Los .npz nuevos guardan todo eso, así
# que se puede "re-correr" y registrar cada configuración aceptada. Las funciones de abajo son
# copias EXACTAS de los bucles de plantillas.buscar e inverso_s7.buscar_margen (no los
# modifican), con un registro agregado. Al final se comprueba que el μ (o el margen) final
# coincide con el guardado: si no coincide, el video avisa.

def _repetir_busqueda(d):
    """Re-corre plantillas.buscar. Devuelve (estados, curva, final): estados = [(it, x, φ, puntaje, estable)]."""
    import plantillas as T
    eps, temp, multa = float(d["eps"]), float(d["temp"]), 1e4
    m_max, iters, esc_min = int(d["m_max"]), int(d["iters"]), float(d["escala_min"])
    rng = np.random.default_rng(int(d["semilla"]))
    x, phi = T.normalizar(np.asarray(d["x0"], float), np.asarray(d["phi0"], float))
    s, fact = T.evaluar(x, phi, eps, multa)
    if s is None:
        s = -1.0
    mejor = s if fact else -1.0
    estados, curva, vistos = [(0, x, phi, s, fact)], [], {}
    for it in range(iters):
        x2, phi2, _ = T.proponer(x, phi, rng, m_max, esc_min)
        if x2 is not None:
            clave = np.round(np.r_[x2, phi2], 12).tobytes()
            if clave in vistos:
                s2, f2 = vistos[clave]
            else:
                s2, f2 = T.evaluar(x2, phi2, eps, multa, umbral=s - 25 * temp)
                if s2 is not None:
                    vistos[clave] = (s2, f2)
            if s2 is not None and s2 >= s - 25 * temp and \
                    (s2 >= s or rng.random() < np.exp((s2 - s) / temp)):
                x, phi, s = x2, phi2, s2
                estados.append((it + 1, x, phi, s, f2))
                if f2 and s > mejor:
                    mejor = s
        curva.append(mejor)
    return estados, curva, mejor


def _repetir_inverso(d):
    """Re-corre inverso_s7.buscar_margen, con el mismo registro."""
    import plantillas as T
    import inverso_s7 as I
    mu_obj, temp = float(d["mu_obj"]), float(d["temp"])
    m_max, iters, esc_min = int(d["m_max"]), int(d["iters"]), float(d["escala_min"])
    rng = np.random.default_rng(int(d["semilla"]))
    x, phi = T.normalizar(np.asarray(d["x0"], float), np.asarray(d["phi0"], float))
    s = I.max_margen(x, phi, mu_obj)[0]
    s = -np.inf if s is None else s
    mejor, estados, curva = s, [(0, x, phi, s, s > 0)], []
    for it in range(iters):
        x2, phi2, _ = T.proponer(x, phi, rng, m_max, esc_min)
        if x2 is not None:
            esc = temp * max(abs(s), 1e-12) if np.isfinite(s) else np.inf
            umbral = s - 25 * esc if np.isfinite(s) else -np.inf
            s2 = I.max_margen(x2, phi2, mu_obj, umbral=umbral)[0]
            if s2 is not None and (s2 >= s or (np.isfinite(s) and s2 >= umbral
                                               and rng.random() < np.exp((s2 - s) / esc))):
                x, phi, s = x2, phi2, s2
                estados.append((it + 1, x, phi, s, s > 0))
                mejor = max(mejor, s)
        curva.append(mejor)
    return estados, curva, mejor


def _cadena_de(ruta):
    """Lista de .npz desde la búsqueda original hasta `ruta`, siguiendo los 'continúa <id>'."""
    import re
    carpeta = os.path.dirname(ruta) or "."
    with open(os.path.join(carpeta, "resultados.csv"), newline="", encoding="utf-8", errors="replace") as f:
        inicio = {r["id"]: r["inicio"] for r in csv.DictReader(f)}
    ids = [os.path.splitext(os.path.basename(ruta))[0]]
    while True:
        m = re.search(r"continúa (\d{4}_\d{6}_\d+)", inicio.get(ids[-1], "").replace("contin\ufffda", "continúa"))
        padre = m.group(1) if m else None
        if not padre or padre in ids or not os.path.exists(os.path.join(carpeta, padre + ".npz")) \
                or "x0" not in np.load(os.path.join(carpeta, padre + ".npz")).files:
            break
        ids.append(padre)
    return [os.path.join(carpeta, i + ".npz") for i in reversed(ids)]


def _colores_por_cuarto(phi):
    import matplotlib.pyplot as plt
    cuarto = np.mod(np.round(phi / (np.pi / 2)), 4).astype(int)
    return plt.cm.tab10(np.array([0, 2, 3, 4])[cuarto])


def video(ruta, n=40, exagerar=40, cuadros=120, fps=12, cadena=False, guardar=None):
    """
    Video de la evolución de una búsqueda: en cada cuadro, la plantilla (izquierda), el grafo
    concreto armado con ella (centro) y la curva del mejor μ o margen (derecha). Los grupos se
    colorean por cuarto (0°, 90°, 180°, 270°), así se ve cómo se parten y se mueven.
    Se guarda como .mp4 si hay ffmpeg instalado; si no, como .gif.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import plantillas as T

    rutas = _cadena_de(ruta) if cadena else [ruta]
    inverso = "mu_obj" in np.load(rutas[-1]).files
    estados, curva, finales, offset = [], [], [], 0
    for r in rutas:
        d = np.load(r)
        if "x0" not in d.files:
            raise ValueError(f"{r} es de una versión anterior: no guardó su punto de partida, "
                             "no se puede re-correr.")
        print(f"Re-corriendo {os.path.basename(r)} ({int(d['iters'])} pasos)...", flush=True)
        e, c, fin = (_repetir_inverso if inverso else _repetir_busqueda)(d)
        guardado = float(d["margen"] if inverso else d["mu"])
        ok = np.isclose(fin, guardado, rtol=0, atol=1e-12) or (not np.isfinite(fin) and not np.isfinite(guardado))
        finales.append((os.path.basename(r), fin, guardado, ok))
        estados += [(it + offset, x, phi, s, f) for it, x, phi, s, f in e]
        curva += c
        offset += int(d["iters"])
    for nombre, fin, guardado, ok in finales:
        print(f"  {nombre}: final re-corrido = {fin:.10g}, guardado = {guardado:.10g}  "
              + ("✓ coincide" if ok else "≠ NO coincide (el video puede no ser fiel)"))

    sel = np.unique(np.linspace(0, len(estados) - 1, min(cuadros, len(estados))).round().astype(int))
    mu_obj = float(np.load(rutas[-1])["mu_obj"]) if inverso else None
    eps = None if inverso else float(np.load(rutas[-1])["eps"])

    def densidades(x, phi):
        if inverso:
            import inverso_s7 as I
            return I.max_margen(x, phi, mu_obj)[2]
        mu, P = T.mejor_densidad(x, phi, eps)
        return P if P is not None else T.puntaje_suave(x, phi, eps)[1]

    print(f"Dibujando {len(sel)} cuadros...", flush=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    its = np.arange(1, len(curva) + 1)
    curva = np.array(curva, float)

    def cuadro(k):
        it, x, phi, s, estable = estados[sel[k]]
        o = np.argsort(np.mod(phi, 2 * np.pi))
        x, phi = np.asarray(x)[o], np.mod(np.asarray(phi)[o], 2 * np.pi)
        P = densidades(x, phi)
        P = np.zeros((len(x), len(x))) if P is None else P
        col = _colores_por_cuarto(phi)
        ang = _posiciones_plantilla(phi, exagerar)
        cx, cy = np.cos(ang), np.sin(ang)
        for ax in (ax1, ax2, ax3):
            ax.clear()
        # plantilla
        ax1.add_patch(plt.Circle((0, 0), 1, fill=False, color="#cfcfcf"))
        for a in range(len(x)):
            for b in range(a + 1, len(x)):
                if P[a, b] > 1e-3:
                    ax1.plot([cx[a], cx[b]], [cy[a], cy[b]], color="#9a9a9a" if P[a, b] > 0.98 else "#eb6834",
                             alpha=0.8, lw=0.5 + 60 * x[a] * x[b] * P[a, b], zorder=1)
        ax1.scatter(cx, cy, s=4000 * x, color=col, edgecolor="white", linewidth=1.5, zorder=2)
        ax1.set_title(f"Plantilla: {len(x)} grupos (separaciones ×{exagerar:g})", fontsize=10)
        # grafo concreto
        if P.any():
            grupo, A = armar_grafo(x, phi, P, n)
            pos = np.zeros((len(grupo), 2))
            radio = 0.06 + 0.25 * np.sqrt(x)
            for a in range(len(x)):
                idx = np.where(grupo == a)[0]
                t = 2 * np.pi * np.arange(len(idx)) / max(len(idx), 1)
                pos[idx, 0] = cx[a] + radio[a] * np.cos(t)
                pos[idx, 1] = cy[a] + radio[a] * np.sin(t)
            ii, jj = np.nonzero(np.triu(A, 1))
            for i, j in zip(ii, jj):
                mismo = grupo[i] == grupo[j]
                ax2.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], lw=0.5, zorder=1,
                         color=col[grupo[i]] if mismo else "#bdbdbd", alpha=0.35 if mismo else 0.1)
            ax2.scatter(pos[:, 0], pos[:, 1], s=24, color=col[grupo], edgecolor="white", linewidth=0.5, zorder=2)
        ax2.set_title(f"Grafo concreto (~{n} vértices)", fontsize=10)
        for ax in (ax1, ax2):
            ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        # curva
        if inverso:
            y = np.where(np.isfinite(curva), np.sign(curva) * np.log10(1 + np.abs(curva) / 1e-12), np.nan)
            ax3.plot(its, y, color="#2a78d6", lw=1.5)
            ax3.axhline(0, color="#52514e", lw=1)
            ax3.set_ylabel("mejor margen  (signo · log10(1 + |m|/1e-12))", fontsize=9)
            ax3.set_title(f"Camino inverso, μ* = {mu_obj}: margen actual = {s:+.3e}", fontsize=10)
        else:
            ok = curva > 0
            ax3.plot(its[ok], curva[ok], color="#2a78d6", lw=1.5)
            ax3.axhline(0.6929077, color="#0b0b0b", ls="--", lw=1)
            ax3.text(its[-1], 0.6929077, "récord ", ha="right", va="bottom", fontsize=8)
            ax3.set_ylabel("mejor μ estable hasta ese paso", fontsize=9)
            ax3.set_title(f"μ actual = {s:.7f}" + ("" if estable else "  (inestable)"), fontsize=10)
        ax3.axvline(it, color="#eb6834", lw=1.2)
        ax3.set_xlabel("iteración" + (" (toda la cadena)" if len(rutas) > 1 else ""), fontsize=9)
        ax3.grid(color="#e4e3df", lw=0.8)
        fig.suptitle(f"{os.path.basename(rutas[-1])}  ·  iteración {it} de {len(curva)}", fontsize=11)
        return []

    anim = animation.FuncAnimation(fig, cuadro, frames=len(sel), blit=False)
    base = os.path.splitext(ruta)[0] + ("_video_cadena" if cadena and len(rutas) > 1 else "_video")
    if guardar is None:
        guardar = base + (".mp4" if animation.writers.is_available("ffmpeg") else ".gif")
    if guardar.endswith(".mp4"):
        anim.save(guardar, writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=90)
    else:
        anim.save(guardar, writer=animation.PillowWriter(fps=fps), dpi=70)
    plt.close(fig)
    return guardar


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dibuja una plantilla guardada como grafo (imagen o video)")
    ap.add_argument("ruta", help="un archivo .npz, o una carpeta de resultados con --mejor")
    ap.add_argument("--mejor", action="store_true", help="usar la mejor plantilla de la carpeta")
    ap.add_argument("--n", type=int, default=40, help="cantidad aproximada de vértices (default 40)")
    ap.add_argument("--exagerar", type=float, default=40, help="factor para ver las mitades separadas")
    ap.add_argument("--video", action="store_true", help="video de la evolución durante la búsqueda")
    ap.add_argument("--cadena", action="store_true",
                    help="con --video: incluir las búsquedas anteriores de las que 'continúa'")
    ap.add_argument("--cuadros", type=int, default=120, help="cuadros del video (default 120)")
    ap.add_argument("--fps", type=int, default=12, help="cuadros por segundo (default 12)")
    a = ap.parse_args()
    ruta = mejor_de_carpeta(a.ruta) if a.mejor else a.ruta
    if a.video:
        print("Video guardado en:", video(ruta, n=a.n, exagerar=a.exagerar, cuadros=a.cuadros,
                                          fps=a.fps, cadena=a.cadena))
    else:
        print("Imagen guardada en:", dibujar(ruta, n=a.n, exagerar=a.exagerar, mostrar=False))
