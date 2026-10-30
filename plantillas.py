"""
Búsqueda de plantillas (templates) de clusters para mejorar la cota inferior de μ_c.

Idea (O'Keeffe 2026, Sec. III): todos los récords son estados de pocos clusters.
En vez de buscar grafos, se buscan plantillas T = (x, φ, P):
    x_a  = fracción de osciladores en el cluster a      (Σ x_a = 1)
    φ_a  = fase común del cluster a
    P_ab = densidad de aristas entre los clusters a y b  (∈ [0, 1])

Una plantilla "estricta" se puede inflar (blow-up) a grafos reales con un estado
estable no sincronizado y conectividad -> μ(T) = min_a d_a. Las condiciones son:
    grado        d_a = Σ_b P_ab x_b                       (queremos maximizar el mínimo)
    equilibrio   F_a = Σ_b P_ab x_b sin(φ_a − φ_b) = 0
    no partirse  c_a = Σ_b P_ab x_b cos(φ_a − φ_b) > 0
    modos        L(y) = Σ_{a<b} x_a x_b P_ab cos(φ_a − φ_b)(y_a − y_b)² > 0  para y ⟂ 1
                 (= λ2 de la matriz M(P) de modos de cluster > 0)

Con x y φ fijos TODO es lineal en P  ->  el problema interno es un LP
(la condición sobre L se impone con planos de corte: cada autovector malo agrega
una desigualdad lineal). El problema externo mueve x y φ, y parte/fusiona clusters.

La condición de modos se impone acá numéricamente con margen ε. Que λ2 > 0 valga
de verdad (aritmética de intervalos + blow-up) es un paso posterior.
"""
import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Problema interno: mejor P para x, φ fijos
# ---------------------------------------------------------------------------
def pares(m):
    return [(a, b) for a in range(m) for b in range(a + 1, m)]


def matriz_modos(x, phi, P):
    """M(P): Laplaciano de los clusters con pesos x_a x_b P_ab cos(φa−φb). L(y) = yᵀMy."""
    W = np.outer(x, x) * P * np.cos(phi[:, None] - phi[None, :])
    np.fill_diagonal(W, 0)
    return np.diag(W.sum(1)) - W


def modos_normalizados(x, phi, P):
    """
    Autovalores de M y = λ diag(x) y  restringido a los modos que NO son la rotación
    global (Σ x_a y_a = 0). Es el Hessiano por oscilador de los modos de cluster:
    rotar el cluster a en y_a mueve x_a·n osciladores, así el margen eps no depende
    del número ni del tamaño de los clusters.
    Devuelve (λ, Y) con λ ordenados (λ[0] = "λ2" de la plantilla) y Σ x_a Y_a² = 1.
    (Ojo: no alcanza con tomar el 2º autovalor de M: si hay uno negativo, la rotación
    global, con autovalor 0, queda en el segundo lugar.)
    """
    x = np.asarray(x, float)
    s = 1 / np.sqrt(x)
    K = s[:, None] * matriz_modos(x, phi, P) * s[None, :]
    # base ortonormal del complemento de sqrt(x) (= rotación global en estas coordenadas)
    Q, _ = np.linalg.qr(np.column_stack([np.sqrt(x), np.eye(len(x))]))
    B = Q[:, 1:len(x)]
    lam, U = np.linalg.eigh(B.T @ K @ B)
    return lam, s[:, None] * (B @ U)


OPCIONES_LP = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}


def _matrices(x, phi):
    """Coeficientes lineales en P de: grado (G), equilibrio (Fs), no partirse (Cc) y modos (w)."""
    m = len(x)
    E = pares(m)
    k = len(E)
    dif = phi[:, None] - phi[None, :]
    S, C = np.sin(dif), np.cos(dif)
    # Matrices "por cluster": fila a, columna e = contribución de P_e
    G = np.zeros((m, k)); Fs = np.zeros((m, k)); Cc = np.zeros((m, k))
    for e, (a, b) in enumerate(E):
        G[a, e], G[b, e] = x[b], x[a]
        Fs[a, e], Fs[b, e] = x[b] * S[a, b], x[a] * S[b, a]
        Cc[a, e], Cc[b, e] = x[b] * C[a, b], x[a] * C[b, a]
    w = np.array([x[a] * x[b] * C[a, b] for a, b in E])     # peso de P_e en L
    return E, G, Fs, Cc, w


def _armar_P(Pv, E, m):
    P = np.eye(m)
    for e, (a, b) in enumerate(E):
        P[a, b] = P[b, a] = Pv[e]
    return P


def _fila_corte(y, E, w):
    """Coeficientes de Σ_e w_e (y_a − y_b)² P_e  (= L(y), lineal en P)."""
    return np.array([w[e] * (y[a] - y[b]) ** 2 for e, (a, b) in enumerate(E)])


def mejor_densidad(x, phi, eps=1e-6, max_cortes=300, umbral=-np.inf):
    """
    max μ  s.a.  d_a ≥ μ,  F_a = 0,  c_a ≥ eps,  λ2(M(P)) ≥ eps,  0 ≤ P ≤ 1,  P_aa = 1.
    Devuelve (μ, P) o (None, None) si no hay plantilla estricta con esos x, φ.
    Garantiza λ2 ≥ eps/2 en la plantilla devuelta.

    Poda: el valor de cada LP de los planos de corte es una COTA SUPERIOR del μ final
    (agregar cortes solo puede bajarlo). Si esa cota ya es < umbral, se corta y se
    devuelve (None, None): es seguro que no hay plantilla estricta con μ ≥ umbral.
    """
    x, phi = np.asarray(x, float), np.asarray(phi, float)
    m = len(x)
    E, G, Fs, Cc, w = _matrices(x, phi)
    k = len(E)                        # variables: P_e para cada par, y μ al final

    # d_a = x_a + G P ≥ μ   ->   −G P + μ ≤ x_a
    A_ub = [np.hstack([-G, np.ones((m, 1))])]
    b_ub = [x.copy()]
    # c_a = x_a + Cc P ≥ eps ->   −Cc P ≤ x_a − eps
    A_ub.append(np.hstack([-Cc, np.zeros((m, 1))]))
    b_ub.append(x - eps)
    A_eq = np.hstack([Fs, np.zeros((m, 1))])
    b_eq = np.zeros(m)
    bounds = [(0, 1)] * k + [(None, None)]
    obj = np.zeros(k + 1); obj[-1] = -1                      # maximizar μ

    cortes = []
    for _ in range(max_cortes):
        A = np.vstack(A_ub + cortes) if cortes else np.vstack(A_ub)
        b = np.concatenate(b_ub + [np.full(len(cortes), -eps)]) if cortes else np.concatenate(b_ub)
        r = linprog(obj, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs",
                    options=OPCIONES_LP)
        if r.status != 0:
            return None, None
        if r.x[-1] < umbral:              # poda: ni siquiera la cota llega al umbral
            return None, None
        P = _armar_P(r.x[:k], E, m)
        lam, Y = modos_normalizados(x, phi, P)
        # Los cortes se piden a nivel eps, pero se acepta λ2 ≥ eps/2: así los planos de
        # corte (método de Kelley) terminan en pocas vueltas en vez de arrastrarse.
        if lam[0] >= eps / 2:
            return r.x[-1], P
        for i in range(m - 1):
            if lam[i] < eps:
                # corte: L(y) ≥ eps  con  y normalizado (Σ x_a y_a² = 1)
                cortes.append(np.append(-_fila_corte(Y[:, i], E, w), 0.0)[None, :])
    return None, None


DESCUENTO = 0.1     # lo que pierde de entrada una configuración sin plantilla estricta


def puntaje_suave(x, phi, eps=1e-6, multa=1e4, max_cortes=300, umbral=-np.inf):
    """
    Para configuraciones SIN plantilla estricta: en vez de −1, devuelve
        max_P [ μ(P) − multa · (eps − λ2(P)) ]  −  DESCUENTO
    o sea "el mejor μ posible, menos una multa por lo que le falta de estabilidad",
    menos un descuento fijo para que nunca le gane a una plantilla estricta de μ
    parecido (C4, con μ = 3/4 y λ2 = 0, puntúa ≈ 0.65 < 0.69).
    La multa (por defecto 1e4) fija cuánto μ está dispuesto a ceder el LP para ganar
    estabilidad: cerca de S7 un λ2 de 1e-6 "cuesta" del orden de 0.01 de μ.
    Configuraciones con λ2 < −1e-3 (muy inestables) no reciben puntaje (None).

    Así la búsqueda sabe si se está acercando a ser estable y puede atravesar zonas
    sin plantilla estricta (por ejemplo, arrancar desde C4).
    Es un solo LP con variables extra s (= λ2 alcanzado, ≤ eps) y v (= faltante ≥ eps − s).
    También tiene poda con la misma lógica que mejor_densidad.
    Devuelve (puntaje, P, λ2 real) o (None, None, None) si se podó o es muy inestable.
    """
    umbral = umbral + DESCUENTO
    x, phi = np.asarray(x, float), np.asarray(phi, float)
    m = len(x)
    E, G, Fs, Cc, w = _matrices(x, phi)
    k = len(E)                        # variables: P (k), μ, s, v
    ceros = lambda filas: np.zeros((filas, 3))
    A_ub = [np.hstack([-G, np.tile([1, 0, 0], (m, 1))]),     # −G P + μ ≤ x_a  (grado ≥ μ)
            np.hstack([-Cc, ceros(m)]),                        # c_a ≥ eps
            np.r_[np.zeros(k), 0, -1, -1][None, :]]          # −s − v ≤ −eps
    b_ub = [x.copy(), x - eps, np.array([-eps])]
    A_eq = np.hstack([Fs, ceros(m)])
    bounds = [(0, 1)] * k + [(None, None), (-1e-3, eps), (0, None)]    # P, μ, s, v
    obj = np.r_[np.zeros(k), -1, 0, multa]                   # maximizar μ − multa·v

    cortes = []
    for _ in range(max_cortes):
        r = linprog(obj, A_ub=np.vstack(A_ub + cortes), b_ub=np.concatenate(b_ub + [np.zeros(len(cortes))]),
                    A_eq=A_eq, b_eq=np.zeros(m), bounds=bounds, method="highs", options=OPCIONES_LP)
        if r.status != 0:
            return None, None, None
        if -r.fun < umbral:               # poda
            return None, None, None
        P = _armar_P(r.x[:k], E, m)
        s = r.x[k + 1]
        lam, Y = modos_normalizados(x, phi, P)
        if lam[0] >= s - eps / 2:         # el λ2 real alcanza lo que el LP prometió
            mu = (P @ x).min()
            return mu - multa * max(0.0, eps - lam[0]) - DESCUENTO, P, lam[0]
        # (el LP pidió λ2 ≥ s; los cortes nuevos corrigen donde el λ2 real no llega)
        for i in range(m - 1):
            if lam[i] < s:
                # corte: L(y) ≥ s   ->   −L(y) + s ≤ 0
                cortes.append(np.r_[-_fila_corte(Y[:, i], E, w), 0, 1, 0][None, :])
    return None, None, None


def evaluar(x, phi, eps=1e-6, multa=1e4, umbral=-np.inf):
    """
    Puntaje de una configuración (x, φ) para la búsqueda:
      - si tiene plantilla estricta: (μ exacto, True)          <- igual que antes
      - si no: (puntaje suave < μ alcanzable, False)
      - si se demuestra que el puntaje es < umbral: (None, False)  <- podada
    """
    mu, _ = mejor_densidad(x, phi, eps, umbral=umbral)
    if mu is not None:
        return mu, True
    p, _, _ = puntaje_suave(x, phi, eps, multa, umbral=umbral)
    return p, False


def verificar(x, phi, P):
    """Chequeo independiente de una plantilla (en punto flotante)."""
    x, phi = np.asarray(x), np.asarray(phi)
    dif = phi[:, None] - phi[None, :]
    d = P @ x
    F = (P * np.sin(dif)) @ x
    c = (P * np.cos(dif)) @ x
    lam, _ = modos_normalizados(x, phi, P)
    return {"mu": float(d.min()), "max|F|": float(np.abs(F).max()),
            "min c": float(c.min()), "lambda2": float(lam[0])}


# ---------------------------------------------------------------------------
# Problema externo: mover tamaños y fases, partir y fusionar clusters
# ---------------------------------------------------------------------------
def normalizar(x, phi):
    x = np.clip(x, 1e-4, None)
    return x / x.sum(), np.mod(phi - phi[0], 2 * np.pi)


def proponer(x, phi, rng, m_max, escala_min=-5):
    """
    Una jugada al azar. Devuelve (x, φ, nombre), o (None, None, nombre) si la jugada no
    cambia nada (partir con m = m_max, fusionar con m = 2): esas no se evalúan.
    """
    m = len(x)
    jugada = rng.choice(["fases", "tamaños", "ambos", "partir", "satélite", "fusionar"],
                        p=[0.3, 0.2, 0.25, 0.1, 0.05, 0.1])
    escala = 10 ** rng.uniform(escala_min, -1)    # perturbaciones en muchas escalas
    x, phi = x.copy(), phi.copy()
    if jugada in ("fases", "ambos"):
        phi += escala * rng.standard_normal(m)
    if jugada in ("tamaños", "ambos"):
        x *= np.exp(escala * rng.standard_normal(m))
    if jugada in ("partir", "satélite"):
        if m >= m_max:
            return None, None, jugada
        a = rng.integers(m)
        h = 10 ** rng.uniform(-4, -1)
        # partir: dos mitades parecidas (como S7); satélite: se desprende un pedazo chico
        f = rng.uniform(0.3, 0.7) if jugada == "partir" else rng.uniform(0.85, 0.97)
        signo = rng.choice([-1, 1])
        x = np.append(x, x[a] * (1 - f)); x[a] *= f
        phi = np.append(phi, phi[a] + signo * h * f); phi[a] -= signo * h * (1 - f)
    if jugada == "fusionar":
        if m <= 2:
            return None, None, jugada
        dd = np.abs(np.angle(np.exp(1j * (phi[:, None] - phi[None, :])))) + 10 * np.eye(m)
        a, b = np.unravel_index(dd.argmin(), dd.shape)
        z = x[a] * np.exp(1j * phi[a]) + x[b] * np.exp(1j * phi[b])
        x[a] += x[b]; phi[a] = np.angle(z)
        x, phi = np.delete(x, b), np.delete(phi, b)
    x, phi = normalizar(x, phi)
    return x, phi, jugada


def buscar(x0, phi0, eps=1e-6, iters=2000, m_max=10, semilla=0, temp=1e-5, log_cada=200,
           historia=None, multa=1e4, estadisticas=None, escala_min=-5):
    """
    Recocido simulado sobre (x, φ, número de clusters). Devuelve (μ, x, φ) de la mejor
    plantilla ESTRICTA encontrada (μ = −1 si nunca encontró ninguna).

    Puntaje: μ exacto si la configuración tiene plantilla estricta; si no, el puntaje
    suave (μ menos una multa por falta de estabilidad). Por eso ahora se puede arrancar
    desde configuraciones no estables, como C4.

    Aceptación: se acepta siempre si mejora; si empeora en Δ, con probabilidad e^(−Δ/temp),
    y NUNCA si empeora más de 25·temp. Gracias a ese tope, antes de evaluar una jugada se
    sabe qué puntaje mínimo necesitaría para tener chance (umbral = actual − 25·temp), y
    la evaluación se corta apenas una cota superior demuestra que no lo alcanza (poda).
    Tampoco se evalúan jugadas que no cambian nada ni configuraciones ya evaluadas.
    """
    rng = np.random.default_rng(semilla)
    x, phi = normalizar(np.asarray(x0, float), np.asarray(phi0, float))
    s, fact = evaluar(x, phi, eps, multa)
    if s is None:
        s = -1.0
    mejor = (s, x, phi) if fact else (-1.0, x, phi)
    vistos = {}
    est = {"evaluadas": 0, "podadas": 0, "repetidas": 0, "nulas": 0, "suaves": 0}
    for it in range(iters):
        x2, phi2, jug = proponer(x, phi, rng, m_max, escala_min)
        if x2 is None:
            est["nulas"] += 1
        else:
            clave = np.round(np.r_[x2, phi2], 12).tobytes()
            if clave in vistos:
                est["repetidas"] += 1
                s2, f2 = vistos[clave]
            else:
                umbral = s - 25 * temp
                s2, f2 = evaluar(x2, phi2, eps, multa, umbral=umbral)
                est["evaluadas"] += 1
                if s2 is None:
                    est["podadas"] += 1    # (no se guarda: con otro umbral podría no podarse)
                else:
                    est["suaves"] += not f2
                    vistos[clave] = (s2, f2)
            if s2 is not None and s2 >= s - 25 * temp and \
                    (s2 >= s or rng.random() < np.exp((s2 - s) / temp)):
                x, phi, s = x2, phi2, s2
                if f2 and s > mejor[0]:
                    mejor = (s, x, phi)
        if historia is not None:          # lista opcional: mejor μ (estricto) en cada iteración
            historia.append(mejor[0])
        if log_cada and it % log_cada == 0:
            print(f"  it {it:5d}  m={len(x):2d}  actual={s:.6f}  mejor={mejor[0]:.6f}  {est}")
    if estadisticas is not None:
        estadisticas.update(est)
    return mejor


# ---------------------------------------------------------------------------
# Plantillas conocidas (para validar)
# ---------------------------------------------------------------------------
def c4():
    return np.full(4, 0.25), np.arange(4) * np.pi / 2


def canale():
    al = np.arctan(0.5)
    phi = np.concatenate([np.arange(4) * np.pi / 2, al + np.arange(4) * np.pi / 2])
    return np.full(8, 1 / 8), phi


Q_STAR, W_STAR = 0.6505691669, 1.3982781340


def s7(h, q=Q_STAR, w=W_STAR):
    a = 1 / (7 + q)
    x = np.array([a] * 6 + [1 - 6 * a])
    phi = np.array([-h, h, np.pi / 2 - w * h, np.pi / 2 + w * h, np.pi - h, np.pi + h, 3 * np.pi / 2])
    return x, phi


def c4_partido(rng, k=None):
    """
    C4 con k de sus 4 grupos partidos en dos (k = 2, 3 o 4 al azar), con separaciones,
    proporciones y ángulos al azar. Generaliza a S7 (que es k = 3 con un patrón preciso):
    sirve como punto de partida para buscar OTRAS deformaciones de C4.
    Casi nunca es estable de entrada; para eso está el puntaje suave.
    """
    x, phi = c4()
    k = int(rng.integers(2, 5)) if k is None else k
    h = 10 ** rng.uniform(-3, -1.5)
    xs, ps = [], []
    partidos = set(rng.choice(4, size=k, replace=False).tolist())
    for a in range(4):
        if a in partidos:
            f = rng.uniform(0.3, 0.7)
            ha = h * rng.uniform(0.5, 2)
            xs += [x[a] * f, x[a] * (1 - f)]
            ps += [phi[a] - ha * (1 - f), phi[a] + ha * f]
        else:
            xs.append(x[a]); ps.append(phi[a])
    return normalizar(np.array(xs), np.array(ps))


def twisted(m, k=1):
    """m clusters iguales equiespaciados con k vueltas (twisted state agrupado)."""
    return np.full(m, 1 / m), 2 * np.pi * k * np.arange(m) / m


def aleatoria(m, rng):
    return normalizar(rng.dirichlet(np.ones(m)), rng.uniform(0, 2 * np.pi, m))


if __name__ == "__main__":
    print("1) Validación del problema interno (μ óptimo con x, φ fijos)")
    casos = [("C4", c4()), ("twisted 12", twisted(12)), ("Canale", canale()),
             ("S7 h=0.005", s7(0.005)), ("S7 h=0.002", s7(0.002))]
    for nombre, (x, phi) in casos:
        for eps in [1e-5, 1e-9]:
            mu, P = mejor_densidad(x, phi, eps)
            txt = "sin plantilla estricta" if P is None else \
                f"μ = {mu:.7f}   λ2 = {verificar(x, phi, P)['lambda2']:.1e}"
            print(f"  {nombre:11s} eps={eps:.0e}  {txt}")
    print(f"  (11/16 = {11/16},  μ0 de O'Keeffe = 0.6929077)")

    print("\n2) Búsqueda externa arrancando de S7 con h = 0.005")
    s, x, phi = buscar(*s7(0.005), eps=1e-9, iters=600, temp=1e-7, log_cada=100)
    print(f"  mejor μ = {s:.7f} con {len(x)} clusters")
    np.savez("mejor_plantilla.npz", x=x, phi=phi, mu=s)
