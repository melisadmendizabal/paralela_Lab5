
import sys
import time

import numpy as np

from imagen import describir, generar_imagen, obtener_imagen_8k


def blur_secuencial(img):
    alto, ancho, canales = img.shape
    salida = np.empty_like(img)
    for y in range(alto):                                   # recorrer filas
        y0, y1 = max(0, y - 1), min(alto - 1, y + 1)
        filas = [img[yy].ravel().tolist() for yy in range(y0, y1 + 1)]   # fila anterior, actual, siguiente
        fila_out = [0] * (ancho * canales)
        for x in range(ancho):                              # recorrer columnas
            x0, x1 = max(0, x - 1), min(ancho - 1, x + 1)
            cuenta = len(filas) * (x1 - x0 + 1)             # 9 dentro, 6 en lados, 4 en esquinas
            for c in range(canales):                        # recorrer canales R, G, B
                suma = 0
                for fila in filas:                          # vecindario 3x3
                    for xx in range(x0, x1 + 1):
                        suma += fila[xx * canales + c]
                fila_out[x * canales + c] = (suma + cuenta // 2) // cuenta
        salida[y] = np.array(fila_out, dtype=np.uint8).reshape(ancho, canales)
    return salida


def blur_numpy(img):
    """Referencia ADICIONAL en CPU (1 hilo) vectorizada con NumPy: suma 9 desplazamientos
    de la imagen rellenada con ceros. Sirve para comparar la GPU contra código de CPU
    que no paga el costo del intérprete de Python. Da exactamente el mismo resultado."""
    alto, ancho, _ = img.shape
    p = np.pad(img.astype(np.uint16), ((1, 1), (1, 1), (0, 0)))
    unos = np.pad(np.ones((alto, ancho), dtype=np.uint16), 1)
    suma = np.zeros((alto, ancho, img.shape[2]), dtype=np.uint16)
    cuenta = np.zeros((alto, ancho), dtype=np.uint16)
    for dy in range(3):
        for dx in range(3):
            suma += p[dy:dy + alto, dx:dx + ancho]
            cuenta += unos[dy:dy + alto, dx:dx + ancho]
    cuenta = cuenta[:, :, None]
    return ((suma + cuenta // 2) // cuenta).astype(np.uint8)


def main():
    if len(sys.argv) == 3:
        img = generar_imagen(int(sys.argv[1]), int(sys.argv[2]))
    else:
        img = obtener_imagen_8k()
    print(describir(img))

    t0 = time.perf_counter()
    salida = blur_secuencial(img)
    t_cpu = time.perf_counter() - t0
    print(f"Tiempo secuencial CPU: {t_cpu:.3f} s")
    print(f"Píxel (1,1) antes: {img[1, 1].tolist()} -> después: {salida[1, 1].tolist()}")
    return t_cpu, salida


if __name__ == "__main__":
    main()
