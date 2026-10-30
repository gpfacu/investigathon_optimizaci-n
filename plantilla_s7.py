"""
plantilla_s7.py — búsquedas que parten SOLO de la familia S7 (O'Keeffe, octubre 2026),
más una verificación en ALTA PRECISIÓN que no depende de los 16 dígitos de la computadora.

No modifica plantillas.py ni busqueda.py: los importa y los usa tal cual.

------------------------------------------------------------------------------------------
Uso
------------------------------------------------------------------------------------------
  1) Búsqueda (mismos parámetros y mismos archivos de salida que busqueda.py):
       python plantilla_s7.py --minutos 300 --iters 3000 --m_max 9 --eps 1e-9 --carpeta corrida_s7

  2) Refinar en alta precisión los mejores resultados de una carpeta:
       python plantilla_s7.py refinar --carpeta corrida_s7 --top 3

  3) Recorrer la familia S7 achicando h, en alta precisión (reproduce el paper):
       python plantilla_s7.py familia

------------------------------------------------------------------------------------------
Qué hace cada parte
------------------------------------------------------------------------------------------
A. Puntos de partida (s7_familia, s7_aleatoria). El paper construye S7 así (Sec. IV y
   Apéndice A): tres grupos de C4 partidos en mitades separadas por ±h, ±w·h, ±h, con
   tamaño de cada mitad a = 1/(7+q) y el grupo entero con 1 − 6a. Su demostración usa
   q = q* − t (un poco por debajo del crítico, para tener margen) y w = w*. Acá cada
   búsqueda arranca de un miembro AL AZAR de esa familia:
       h  entre 0.001 y 0.05   (escala logarítmica)
       t  entre 0 y 0.05        -> q = q* − t
       w  = w* · (1 ± 2%)
       mitades iguales (como el paper) o levemente desiguales (45/55 a 55/45)
   Es lo mismo que hacía c4_partido, pero sorteando dentro de la familia del paper en
   lugar de partir C4 a ciegas.

B. Lanzador. Reemplaza, solo mientras corre este programa, la función que elige los
   puntos de partida en busqueda.py, y después llama a busqueda.main(). Todo lo demás
   (procesos en paralelo, poda, puntos agotados, archivos csv/txt/npz, mejor.txt) es
   exactamente lo de busqueda.py. ver_resultados.ipynb sirve igual: la familia aparece
   como "s7_paper".

C. Alta precisión (mejor_densidad_alta). El mismo cálculo que plantillas.mejor_densidad
   (maximizar μ con equilibrio, c > 0 y λ2 ≥ eps, por planos de corte), pero hecho con el
   módulo decimal de Python, que trabaja con la cantidad de dígitos que se pida (por
   defecto 50). Incluye su propio simplex, sus propios senos y cosenos y su propio
   cálculo de autovalores (Jacobi). No necesita instalar nada.
   Es unas 100 veces más lento que el LP normal: no sirve para buscar, sirve para tomar
   las mejores plantillas y ver hasta dónde llegan con márgenes eps de 1e-12, 1e-15, ...,
   que el LP normal no puede distinguir del cero.
"""
import os
import re
import sys
from decimal import Decimal, getcontext, localcontext

import numpy as np
import plantillas as T
import busqueda as B

Q_STAR, W_STAR = T.Q_STAR, T.W_STAR
MU0 = 0.6929077063695863          # (4 + 2q*)/(7 + q*): el récord del paper


# ======================================================================================
# A. La familia S7 y los arranques al azar
# ======================================================================================
def s7_familia(h, t=0.0, w=W_STAR, f=0.5):
    """
    Un miembro de la familia S7 del paper.
      h : separación de las mitades (radianes)
      t : q = q* − t   (el paper usa 0 < t ≤ 0.05)
      w : cuánto más separadas están las mitades del cuarto de 90° (paper: w* ≈ 1.398)
      f : fracción de cada grupo partido que va a la mitad "de atrás" (0.5 = mitades iguales)
    Devuelve (x, φ) en el orden Am, Ap, Bm, Bp, Cm, Cp, W.
    Con f ≠ 0.5 la separación total sigue siendo 2h (o 2wh) y el centro de masa de cada
    cuarto sigue en su lugar.
    """
    q = Q_STAR - t
    a = 1.0 / (7.0 + q)
    x = np.array([2 * a * f, 2 * a * (1 - f)] * 3 + [1 - 6 * a])
    atras, adelante = -2 * (1 - f), 2 * f            # = −1 y +1 con mitades iguales
    phi = np.array([atras * h, adelante * h,
                    np.pi / 2 + atras * w * h, np.pi / 2 + adelante * w * h,
                    np.pi + atras * h, np.pi + adelante * h,
                    3 * np.pi / 2])
    return x, phi


def s7_aleatoria(rng):
    """Un miembro al azar de la familia S7 (ver el encabezado). Devuelve (descripción, x, φ)."""
    h = 10 ** rng.uniform(-3, -1.3)
    t = rng.uniform(0, 0.05)
    w = W_STAR * (1 + rng.uniform(-0.02, 0.02))
    f = 0.5 if rng.random() < 0.5 else rng.uniform(0.45, 0.55)
    x, phi = s7_familia(h, t, w, f)
    return f"s7(h={h:.4f},t={t:.3f},w={w:.3f},f={f:.2f})", x, phi


# ======================================================================================
# B. El lanzador: busqueda.py con puntos de partida de la familia S7
# ======================================================================================
FAMILIA = "s7_paper"


def elegir_inicio_s7(rng, carpeta):
    """
    Reemplazo de busqueda.elegir_inicio. Misma lógica que el original:
      - 40% de las veces sigue desde uno de los 3 mejores resultados DISTINTOS de esta
        familia que no estén agotados (busqueda.agotados);
      - el resto, arranca de un miembro nuevo al azar de la familia S7.
    """
    filas = [r for r in B.leer_csv(carpeta) if float(r["mu"]) > 0 and r.get("familia") == FAMILIA]
    fuera = B.agotados(filas)
    candidatas = [r for r in filas if r["id"] not in fuera
                  and os.path.exists(os.path.join(carpeta, r["id"] + ".npz"))]
    if candidatas and rng.random() < 0.4:
        candidatas.sort(key=lambda r: -float(r["mu"]))
        distintas, vistos = [], set()
        for r in candidatas:
            if round(float(r["mu"]), 8) not in vistos:
                vistos.add(round(float(r["mu"]), 8)); distintas.append(r)
        r = distintas[rng.integers(min(3, len(distintas)))]
        d = np.load(os.path.join(carpeta, r["id"] + ".npz"))
        return f"continúa {r['id']}", FAMILIA, d["x"], d["phi"]
    desc, x, phi = s7_aleatoria(rng)
    return desc, FAMILIA, x, phi


def lanzar_busqueda():
    """Corre busqueda.main() con los arranques de la familia S7 (mismos argumentos)."""
    B.elegir_inicio = elegir_inicio_s7
    print("Puntos de partida: familia S7 del paper (y continuaciones de sus mejores resultados)")
    B.main()


# ======================================================================================
# C. Alta precisión con el módulo decimal
# ======================================================================================
# --- funciones elementales en Decimal --------------------------------------------------
def _pi():
    """π con la precisión actual (receta de la documentación de Python)."""
    getcontext().prec += 2
    three = Decimal(3)
    lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
    while s != lasts:
        lasts = s
        n, na = n + na, na + 8
        d, da = d + da, da + 32
        t = (t * n) / d
        s += t
    getcontext().prec -= 2
    return +s


def _cos_sin(x, pi):
    """(cos x, sin x) por series de Taylor, después de llevar x a [−π, π]."""
    dos_pi = 2 * pi
    x = x - dos_pi * (x / dos_pi).to_integral_value()
    getcontext().prec += 2
    c, s = Decimal(1), x
    term_c, term_s, i = Decimal(1), x, 1
    while True:
        term_c = -term_c * x * x / ((2 * i - 1) * (2 * i))
        term_s = -term_s * x * x / ((2 * i) * (2 * i + 1))
        c2, s2 = c + term_c, s + term_s
        if c2 == c and s2 == s:
            break
        c, s, i = c2, s2, i + 1
    getcontext().prec -= 2
    return +c, +s


# --- simplex de dos fases en Decimal ---------------------------------------------------
def _simplex_max(filas, c, tol):
    """
    Maximiza c·v sujeto a filas, con v ≥ 0. Cada fila es (coeficientes, sentido, lado_derecho)
    con sentido "<=" o "=". Simplex de tablero denso con dos fases y regla de Bland (que
    garantiza que no cicla). Devuelve (valor, v) o (None, None) si no hay solución.
    """
    n = len(c)
    filas_std = []
    for a, sentido, b in filas:
        a = list(a)
        if b < 0:
            a, b = [-v for v in a], -b
            sentido = {"<=": ">=", "=": "="}[sentido]
        filas_std.append((a, sentido, b))
    nh = sum(1 for _, s_, _ in filas_std if s_ != "=")              # holguras
    na = sum(1 for _, s_, _ in filas_std if s_ != "<=")             # artificiales
    ncol = n + nh + na
    tab, base = [], []
    ih, ia = n, n + nh
    for a, sentido, b in filas_std:
        fila = a + [Decimal(0)] * (nh + na) + [b]
        if sentido == "<=":
            fila[ih] = Decimal(1); base.append(ih); ih += 1
        elif sentido == ">=":
            fila[ih] = Decimal(-1); ih += 1
            fila[ia] = Decimal(1); base.append(ia); ia += 1
        else:
            fila[ia] = Decimal(1); base.append(ia); ia += 1
        tab.append(fila)
    artificiales = set(range(n + nh, ncol))

    def iterar(costo):
        # fila z: z_j = c_B·B⁻¹A_j − c_j  (para maximizar: entra la primera j con z_j < 0)
        z = [-costo[j] for j in range(ncol)] + [Decimal(0)]
        for i, bi in enumerate(base):
            if costo[bi] != 0:
                z = [zj + costo[bi] * tij for zj, tij in zip(z, tab[i])]
        while True:
            entra = next((j for j in range(ncol) if z[j] < -tol and j not in prohibidas), None)
            if entra is None:
                return z[-1]
            sale, mejor = None, None
            for i, fila in enumerate(tab):
                if fila[entra] > tol:
                    r = fila[-1] / fila[entra]
                    if mejor is None or r < mejor - tol or (abs(r - mejor) <= tol and base[i] < base[sale]):
                        sale, mejor = i, r
            if sale is None:
                raise RuntimeError("LP no acotado")
            piv = tab[sale][entra]
            tab[sale] = [v / piv for v in tab[sale]]
            for i in range(len(tab)):
                if i != sale and tab[i][entra] != 0:
                    k = tab[i][entra]
                    tab[i] = [u - k * v for u, v in zip(tab[i], tab[sale])]
            if z[entra] != 0:
                k = z[entra]
                z = [u - k * v for u, v in zip(z, tab[sale])]
            base[sale] = entra

    prohibidas = set()
    if artificiales:
        costo1 = [Decimal(0)] * ncol
        for j in artificiales:
            costo1[j] = Decimal(-1)
        if iterar(costo1) < -tol * 100:
            return None, None
        # sacar de la base las artificiales que quedaron en cero
        for i in range(len(tab)):
            if base[i] in artificiales:
                j = next((j for j in range(n + nh) if abs(tab[i][j]) > tol), None)
                if j is not None:
                    piv = tab[i][j]
                    tab[i] = [v / piv for v in tab[i]]
                    for k in range(len(tab)):
                        if k != i and tab[k][j] != 0:
                            f = tab[k][j]
                            tab[k] = [u - f * v for u, v in zip(tab[k], tab[i])]
                    base[i] = j
        prohibidas = artificiales
    costo2 = list(c) + [Decimal(0)] * (ncol - n)
    valor = iterar(costo2)
    v = [Decimal(0)] * ncol
    for i, bi in enumerate(base):
        v[bi] = tab[i][-1]
    return valor, v[:n]


# --- autovalores de una matriz simétrica (Jacobi) ---------------------------------------
def _jacobi(A, tol):
    """Autovalores y autovectores (columnas de V) de A simétrica, por rotaciones de Jacobi."""
    n = len(A)
    A = [fila[:] for fila in A]
    V = [[Decimal(int(i == j)) for j in range(n)] for i in range(n)]
    for _ in range(100):
        off = sum(A[i][j] * A[i][j] for i in range(n) for j in range(i + 1, n))
        if off < tol * tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(A[p][q]) <= tol * tol:
                    continue
                theta = (A[q][q] - A[p][p]) / (2 * A[p][q])
                t = (1 if theta >= 0 else -1) / (abs(theta) + (theta * theta + 1).sqrt())
                c = 1 / (t * t + 1).sqrt()
                s = t * c
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p], A[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k], A[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p], V[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [A[i][i] for i in range(n)], V


def _modos(x, cosd, P, tol):
    """
    Igual que plantillas.modos_normalizados, en Decimal: autovalores de M y = λ diag(x) y
    sin la rotación global, y sus vectores y (normalizados con Σ x y² = 1).
    """
    m = len(x)
    W = [[x[a] * x[b] * P[a][b] * cosd[a][b] if a != b else Decimal(0) for b in range(m)] for a in range(m)]
    M = [[(sum(W[a]) if a == b else -W[a][b]) for b in range(m)] for a in range(m)]
    raiz = [xa.sqrt() for xa in x]
    K = [[M[a][b] / (raiz[a] * raiz[b]) for b in range(m)] for a in range(m)]
    # base ortonormal del complemento de sqrt(x) (Gram-Schmidt)
    v0 = raiz
    base = [v0]
    for e in range(m):
        v = [Decimal(int(i == e)) for i in range(m)]
        for u in base:
            pr = sum(a * b for a, b in zip(v, u))
            v = [a - pr * b for a, b in zip(v, u)]
        nv = sum(a * a for a in v).sqrt()
        if nv > Decimal("1e-6"):
            base.append([a / nv for a in v])
        if len(base) == m:
            break
    Bm = base[1:]                                         # m − 1 vectores
    Kr = [[sum(Bm[i][a] * sum(K[a][b] * Bm[j][b] for b in range(m)) for a in range(m))
           for j in range(m - 1)] for i in range(m - 1)]
    lam, U = _jacobi(Kr, tol)
    orden = sorted(range(m - 1), key=lambda i: lam[i])
    Y = []
    for i in orden:
        z = [sum(Bm[k][a] * U[k][i] for k in range(m - 1)) for a in range(m)]
        Y.append([z[a] / raiz[a] for a in range(m)])
    return [lam[i] for i in orden], Y


def mejor_densidad_alta(x, phi, eps, digitos=60, max_cortes=200):
    """
    El mismo problema que plantillas.mejor_densidad, en aritmética de `digitos` dígitos:
        máx μ  s.a.  d_a ≥ μ,  F_a = 0,  c_a ≥ eps,  λ2 ≥ eps,  0 ≤ P ≤ 1
    con planos de corte para λ2 (se acepta λ2 ≥ eps/2, igual que el original).
    x y φ entran como números de la computadora y se toman como exactos.
    Devuelve dict con mu, lambda2, max|F|, P (todo en Decimal), o None si no hay plantilla.
    """
    with localcontext() as ctx:
        ctx.prec = digitos
        # tolerancia para "es cero": la mitad de los dígitos. Con h chico las cuentas del simplex
        # amplifican el redondeo (del orden de 1/h²), así que no conviene una tolerancia más fina.
        tol = Decimal(10) ** (-(digitos // 2))
        pi = _pi()
        m = len(x)
        xd = [Decimal(float(v)) for v in x]
        sd = sum(xd)
        xd = [v / sd for v in xd]
        ph = [Decimal(float(v)) for v in phi]
        cosd = [[None] * m for _ in range(m)]
        sind = [[None] * m for _ in range(m)]
        for a in range(m):
            for b in range(m):
                cosd[a][b], sind[a][b] = _cos_sin(ph[a] - ph[b], pi)
        epsd = Decimal(repr(float(eps)))
        pares = [(a, b) for a in range(m) for b in range(a + 1, m)]
        k = len(pares)
        n = k + 1                                         # P_e y μ (al final)
        filas = []
        for a in range(m):                                # grado: −Σ P x_b + μ ≤ x_a
            co = [Decimal(0)] * n; co[-1] = Decimal(1)
            for e, (u, v) in enumerate(pares):
                if a == u: co[e] = -xd[v]
                if a == v: co[e] = -xd[u]
            filas.append((co, "<=", xd[a]))
        for a in range(m):                                # c_a ≥ eps
            co = [Decimal(0)] * n
            for e, (u, v) in enumerate(pares):
                if a == u: co[e] = -xd[v] * cosd[u][v]
                if a == v: co[e] = -xd[u] * cosd[v][u]
            filas.append((co, "<=", xd[a] - epsd))
        for e in range(k):                                # P ≤ 1
            co = [Decimal(0)] * n; co[e] = Decimal(1)
            filas.append((co, "<=", Decimal(1)))
        for a in range(m):                                # equilibrio F_a = 0
            co = [Decimal(0)] * n
            for e, (u, v) in enumerate(pares):
                if a == u: co[e] = xd[v] * sind[u][v]
                if a == v: co[e] = xd[u] * sind[v][u]
            filas.append((co, "=", Decimal(0)))
        objetivo = [Decimal(0)] * k + [Decimal(1)]
        wcoef = [xd[u] * xd[v] * cosd[u][v] for u, v in pares]
        cortes = []
        for _ in range(max_cortes):
            mu, sol = _simplex_max(filas + cortes, objetivo, tol)
            if mu is None:
                return None
            P = [[Decimal(1) if a == b else Decimal(0) for b in range(m)] for a in range(m)]
            for e, (u, v) in enumerate(pares):
                P[u][v] = P[v][u] = sol[e]
            lam, Y = _modos(xd, cosd, P, tol)
            if lam[0] >= epsd / 2:
                F = [abs(sum(P[a][b] * xd[b] * sind[a][b] for b in range(m))) for a in range(m)]
                return {"mu": mu, "lambda2": lam[0], "max|F|": max(F), "P": P}
            for i, li in enumerate(lam):
                if li < epsd:
                    y = Y[i]
                    co = [-wcoef[e] * (y[u] - y[v]) ** 2 for e, (u, v) in enumerate(pares)] + [Decimal(0)]
                    cortes.append((co, "<=", -epsd))
        return None


# --- los dos usos -----------------------------------------------------------------------
def _ordenar_por_fase(x, phi):
    o = np.argsort(np.mod(phi, 2 * np.pi))
    return np.asarray(x)[o], np.mod(np.asarray(phi)[o], 2 * np.pi)


def refinar(carpeta, top=3, epsilones=(1e-9, 1e-12, 1e-15), digitos=60):
    """
    Toma las `top` mejores plantillas distintas de la carpeta y recalcula su μ óptimo en alta
    precisión con márgenes cada vez más chicos. Con eps = 1e-9 tiene que coincidir con lo
    que dio la búsqueda (control); con eps más chicos muestra cuánto más da esa misma
    configuración de grupos cuando se le exige menos margen.
    """
    filas = sorted([r for r in B.leer_csv(carpeta) if float(r["mu"]) > 0], key=lambda r: -float(r["mu"]))
    elegidas, vistos = [], set()
    for r in filas:
        if round(float(r["mu"]), 8) not in vistos:
            vistos.add(round(float(r["mu"]), 8)); elegidas.append(r)
        if len(elegidas) == top:
            break
    print(f"Refinamiento en alta precisión ({digitos} dígitos) de las {len(elegidas)} mejores de '{carpeta}'")
    print(f"Referencia: récord del paper μ0 = {MU0:.13f}\n")
    for r in elegidas:
        d = np.load(os.path.join(carpeta, r["id"] + ".npz"))
        x, phi = _ordenar_por_fase(d["x"], d["phi"])
        print(f"{r['id']}  (μ de la búsqueda = {float(r['mu']):.10f}, {len(x)} grupos)")
        for eps in epsilones:
            z = mejor_densidad_alta(x, phi, eps, digitos)
            if z is None:
                print(f"   eps = {eps:.0e}:  sin plantilla estricta")
            else:
                print(f"   eps = {eps:.0e}:  μ = {float(z['mu']):.13f}   λ2 = {float(z['lambda2']):.2e}"
                      f"   |F| = {float(z['max|F|']):.0e}")
        print()


def recorrer_familia(digitos=80):
    """
    La familia S7 del paper con h cada vez más chico, en alta precisión.
    El margen de estabilidad de S7 es del orden de t·h² (paper, Sec. VI), así que para cada
    (h, t) se exige eps = 0.1·t·h². Muestra:
      - cómo μ se acerca a μ0 desde abajo cuando h y t bajan (nunca lo supera);
      - que el margen necesario cae como h²: con h = 1e-6 ya es ~1e-19, imposible de ver
        con los 16 dígitos de la computadora;
      - la última fila es el miembro que certifica el paper (h = 1e-6, t = 1e-5):
        μ* = 0.6929050246 con margen 6.2e-19.
    """
    print(f"Familia S7 del paper en alta precisión ({digitos} dígitos), w = w*, eps = 0.1·t·h²")
    print(f"Referencia: μ0 = {MU0:.13f}\n")
    print(f"{'h':>7} {'t':>7} {'eps':>9}  {'μ':>17}  {'μ0 − μ':>9}  {'λ2':>9}  {'max|F|':>8}")
    for h, t in [(1e-2, 1e-3), (1e-3, 1e-3), (1e-4, 1e-3), (1e-5, 1e-3), (1e-6, 1e-5)]:
        x, phi = _ordenar_por_fase(*s7_familia(h, t=t))
        eps = 0.1 * t * h * h
        z = mejor_densidad_alta(x, phi, eps, digitos)
        if z is None:
            print(f"{h:7.0e} {t:7.0e} {eps:9.1e}  sin plantilla estricta")
            continue
        mu = float(z["mu"])
        print(f"{h:7.0e} {t:7.0e} {eps:9.1e}  {mu:17.13f}  {MU0 - mu:9.2e}  "
              f"{float(z['lambda2']):9.2e}  {float(z['max|F|']):8.0e}")
    print("\n(el paper certifica, para h = 1e-6 y t = 1e-5: μ* = 0.6929050246 con margen 6.23e-19)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "refinar":
        import argparse
        ap = argparse.ArgumentParser(prog="plantilla_s7.py refinar")
        ap.add_argument("--carpeta", default="resultados")
        ap.add_argument("--top", type=int, default=3)
        ap.add_argument("--digitos", type=int, default=60)
        a = ap.parse_args(sys.argv[2:])
        refinar(a.carpeta, a.top, digitos=a.digitos)
    elif len(sys.argv) > 1 and sys.argv[1] == "familia":
        recorrer_familia()
    else:
        lanzar_busqueda()
