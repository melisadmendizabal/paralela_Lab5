
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

CX = 5.0                                   # columna principal
COLOR_PROCESO = "#dbe8f6"
COLOR_DECISION = "#fdf0c8"
COLOR_ES = "#d9f0dc"                       # entrada / salida
COLOR_TERMINAL = "#e9e3f5"
BORDE = "#33415c"
FUENTE = 8.5

fig, ax = plt.subplots(figsize=(8.2, 11.3))
ax.set_xlim(0, 13)
ax.set_ylim(10.0, 30.9)
ax.axis("off")


def texto(x, y, t, **kw):
    ax.text(x, y, t, ha="center", va="center", fontsize=kw.pop("fs", FUENTE), color="#1b1f29",
            linespacing=1.35, **kw)


def proceso(x, y, t, w=3.6, h=0.9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="square,pad=0",
                                fc=COLOR_PROCESO, ec=BORDE, lw=1.1))
    texto(x, y, t)
    return {"x": x, "y": y, "w": w, "h": h}


def terminal(x, y, t, w=2.2, h=0.8):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                fc=COLOR_TERMINAL, ec=BORDE, lw=1.1))
    texto(x, y, t, fontweight="bold")
    return {"x": x, "y": y, "w": w, "h": h}


def entrada_salida(x, y, t, w=4.6, h=1.1):
    s = 0.3
    ax.add_patch(Polygon([(x - w / 2 + s, y + h / 2), (x + w / 2 + s, y + h / 2),
                          (x + w / 2 - s, y - h / 2), (x - w / 2 - s, y - h / 2)],
                         closed=True, fc=COLOR_ES, ec=BORDE, lw=1.1))
    texto(x, y, t)
    return {"x": x, "y": y, "w": w, "h": h}


def decision(x, y, t, w=3.2, h=1.5):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                         closed=True, fc=COLOR_DECISION, ec=BORDE, lw=1.1))
    texto(x, y, t, fontweight="bold")
    return {"x": x, "y": y, "w": w, "h": h}


def flecha(puntos, etiqueta=None, pos_etiqueta=None):
    """Polilínea con punta de flecha en el último segmento."""
    xs, ys = zip(*puntos)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=BORDE, lw=1.1, solid_joinstyle="miter")
    ax.annotate("", xy=puntos[-1], xytext=puntos[-2],
                arrowprops=dict(arrowstyle="-|>", color=BORDE, lw=1.1, mutation_scale=11,
                                shrinkA=0, shrinkB=0))
    if etiqueta:
        ax.text(*pos_etiqueta, etiqueta, fontsize=FUENTE - 0.5, color="#9a3412", fontweight="bold",
                ha="center", va="center")


def arriba(n): return n["y"] + n["h"] / 2
def abajo(n): return n["y"] - n["h"] / 2
def izq(n): return n["x"] - n["w"] / 2
def der(n): return n["x"] + n["w"] / 2


# ---- nodos de la columna principal ----
inicio = terminal(CX, 30.3, "Inicio")
entrada = entrada_salida(CX, 28.75, "Entrada: imagen img[alto][ancho][3]\n(uint8 RGB, alto = 4320, ancho = 7680)", w=5.9)
init = proceso(CX, 27.15, "salida ← imagen vacía (mismo tamaño)\ny ← 0", w=5.0)
J_Y = 26.0
dec_y = decision(CX, 24.9, "¿y < alto?")
init_x = proceso(CX, 23.3, "x ← 0", w=2.2, h=0.7)
J_X = 22.3
dec_x = decision(CX, 21.2, "¿x < ancho?")
vecindario = proceso(CX, 19.15, "Delimitar vecindario 3×3 dentro de la imagen:\n"
                                "y0 = máx(0, y−1),  y1 = mín(alto−1, y+1)\n"
                                "x0 = máx(0, x−1),  x1 = mín(ancho−1, x+1)\n"
                                "cuenta = (y1−y0+1)·(x1−x0+1)   → 9, 6 o 4", w=6.2, h=1.75)
init_c = proceso(CX, 17.35, "c ← 0   (canal R)", w=2.8, h=0.7)
J_C = 16.35
dec_c = decision(CX, 15.25, "¿c < 3?")
suma = proceso(CX, 13.5, "suma ← Σ img[yy][xx][c]\npara yy ∈ [y0, y1],  xx ∈ [x0, x1]", w=4.6, h=1.0)
prom = proceso(CX, 12.0, "salida[y][x][c] ← redondear(suma / cuenta)\n= (suma + cuenta // 2) // cuenta", w=5.4, h=1.0)
inc_c = proceso(CX, 10.55, "c ← c + 1", w=2.4, h=0.7)

# ---- nodos laterales ----
salida = entrada_salida(10.3, 24.9, "Salida: imagen\ndesenfocada\nsalida[alto][ancho][3]", w=3.3, h=1.5)
fin = terminal(10.3, 23.0, "Fin", w=1.8, h=0.7)
inc_y = proceso(1.55, 21.2, "y ← y + 1", w=2.1, h=0.7)
inc_x = proceso(9.4, 15.25, "x ← x + 1", w=2.1, h=0.7)

# ---- flechas de la columna principal ----
flecha([(CX, abajo(inicio)), (CX, arriba(entrada))])
flecha([(CX, abajo(entrada)), (CX, arriba(init))])
flecha([(CX, abajo(init)), (CX, arriba(dec_y))])
flecha([(CX, abajo(dec_y)), (CX, arriba(init_x))], "Sí", (CX + 0.35, abajo(dec_y) - 0.2))
flecha([(CX, abajo(init_x)), (CX, arriba(dec_x))])
flecha([(CX, abajo(dec_x)), (CX, arriba(vecindario))], "Sí", (CX + 0.35, abajo(dec_x) - 0.2))
flecha([(CX, abajo(vecindario)), (CX, arriba(init_c))])
flecha([(CX, abajo(init_c)), (CX, arriba(dec_c))])
flecha([(CX, abajo(dec_c)), (CX, arriba(suma))], "Sí", (CX + 0.35, abajo(dec_c) - 0.2))
flecha([(CX, abajo(suma)), (CX, arriba(prom))])
flecha([(CX, abajo(prom)), (CX, arriba(inc_c))])

# ---- ramas "No" y ciclos ----
flecha([(der(dec_y), dec_y["y"]), (izq(salida), salida["y"])], "No", (der(dec_y) + 0.45, dec_y["y"] + 0.25))
flecha([(salida["x"], abajo(salida)), (fin["x"], arriba(fin))])

flecha([(izq(dec_x), dec_x["y"]), (der(inc_y), inc_y["y"])], "No", (izq(dec_x) - 0.45, dec_x["y"] + 0.25))
flecha([(inc_y["x"], arriba(inc_y)), (inc_y["x"], J_Y), (CX, J_Y)])            # siguiente fila

flecha([(der(dec_c), dec_c["y"]), (izq(inc_x), inc_x["y"])], "No", (der(dec_c) + 0.45, dec_c["y"] + 0.25))
flecha([(der(inc_x), inc_x["y"]), (11.6, inc_x["y"]), (11.6, J_X), (CX, J_X)])   # siguiente columna

flecha([(izq(inc_c), inc_c["y"]), (0.8, inc_c["y"]), (0.8, J_C), (CX, J_C)])    # siguiente canal

ax.text(0.95, J_C + 0.22, "siguiente canal", fontsize=FUENTE - 1, color="#555", ha="left", style="italic")
ax.text(6.9, J_X + 0.22, "siguiente columna", fontsize=FUENTE - 1, color="#555", ha="left", style="italic")
ax.text(1.75, J_Y + 0.22, "siguiente fila", fontsize=FUENTE - 1, color="#555", ha="left", style="italic")

fig.savefig("diagrama_flujo.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Guardado diagrama_flujo.png")
