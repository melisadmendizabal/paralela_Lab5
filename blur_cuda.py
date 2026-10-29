
import math
import sys
import time
import warnings

import numpy as np
from numba import cuda, int32, uint8

from imagen import describir, generar_imagen, obtener_imagen_8k

BLOQUE_X = 16                     # hilos por bloque en x (columnas)
BLOQUE_Y = 16                     # hilos por bloque en y (filas) -> 256 hilos por bloque
TILE_X = BLOQUE_X + 2             # bloque + halo de 1 píxel a cada lado
TILE_Y = BLOQUE_Y + 2
CANALES = 3


@cuda.jit
def blur_global(img, salida):
    x, y = cuda.grid(2)                                   # columna y fila del píxel de este hilo
    alto, ancho = img.shape[0], img.shape[1]
    if x < ancho and y < alto:                            # hilos fuera de la imagen no hacen nada
        y0, y1 = max(0, y - 1), min(alto - 1, y + 1)
        x0, x1 = max(0, x - 1), min(ancho - 1, x + 1)
        cuenta = int32((y1 - y0 + 1) * (x1 - x0 + 1))
        for c in range(CANALES):
            suma = int32(0)
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    suma += int32(img[yy, xx, c])
            salida[y, x, c] = uint8((suma + cuenta // 2) // cuenta)


@cuda.jit
def blur_shared(img, salida):
    tile = cuda.shared.array((TILE_Y, TILE_X, CANALES), dtype=uint8)
    alto, ancho = img.shape[0], img.shape[1]
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    origen_x = cuda.blockIdx.x * BLOQUE_X - 1             # esquina superior izquierda del tile
    origen_y = cuda.blockIdx.y * BLOQUE_Y - 1

    # 1) Carga cooperativa: los 256 hilos copian los 18x18 píxeles del tile (324 > 256,
    #    así que algunos hilos copian dos). Fuera de la imagen se guarda 0 para que no sume.
    for k in range(ty * BLOQUE_X + tx, TILE_X * TILE_Y, BLOQUE_X * BLOQUE_Y):
        sy, sx = k // TILE_X, k % TILE_X
        gy, gx = origen_y + sy, origen_x + sx
        dentro = gy >= 0 and gy < alto and gx >= 0 and gx < ancho
        for c in range(CANALES):
            tile[sy, sx, c] = img[gy, gx, c] if dentro else uint8(0)
    cuda.syncthreads()                                    # esperar a que el tile esté completo

    # 2) Cada hilo calcula su píxel leyendo de memoria compartida. Gracias al relleno con 0,
    #    el vecindario siempre es un 3x3 fijo (el compilador lo desenrolla) y solo cambia
    #    el divisor en los bordes; en el interior se divide entre la constante 9.
    x, y = cuda.grid(2)
    if x < ancho and y < alto:
        cuenta = int32((min(alto - 1, y + 1) - max(0, y - 1) + 1) *
                       (min(ancho - 1, x + 1) - max(0, x - 1) + 1))
        for c in range(CANALES):
            suma = int32(0)
            for dy in range(3):
                for dx in range(3):
                    suma += int32(tile[ty + dy, tx + dx, c])
            if cuenta == 9:
                salida[y, x, c] = uint8((suma + 4) // 9)
            else:
                salida[y, x, c] = uint8((suma + cuenta // 2) // cuenta)


def configurar_grid(alto, ancho):
    """Grid 2D: un hilo por píxel, bloques de 16x16 (x = columnas, y = filas)."""
    bloques = (math.ceil(ancho / BLOQUE_X), math.ceil(alto / BLOQUE_Y))
    return bloques, (BLOQUE_X, BLOQUE_Y)


def blur_cuda(img, kernel="shared", salida_host=None):
    """Aplica el blur en GPU midiendo cada fase. Devuelve (imagen, tiempos).

    salida_host permite pasar un arreglo destino (por ejemplo, memoria pinned).
    """
    k = blur_shared if kernel == "shared" else blur_global
    bloques, hilos = configurar_grid(img.shape[0], img.shape[1])

    # 1) Host -> Device (incluye reservar el arreglo de salida en la VRAM)
    cuda.synchronize()
    t0 = time.perf_counter()
    d_img = cuda.to_device(img)
    d_salida = cuda.device_array_like(img)
    cuda.synchronize()
    t_h2d = time.perf_counter() - t0

    # 2) Kernel (la ejecución es asíncrona: hay que sincronizar antes de detener el reloj)
    t0 = time.perf_counter()
    k[bloques, hilos](d_img, d_salida)
    cuda.synchronize()
    t_kernel = time.perf_counter() - t0

    # 3) Device -> Host
    t0 = time.perf_counter()
    salida = d_salida.copy_to_host(salida_host) if salida_host is not None else d_salida.copy_to_host()
    cuda.synchronize()
    t_d2h = time.perf_counter() - t0

    del d_img, d_salida
    tiempos = {"h2d": t_h2d, "kernel": t_kernel, "d2h": t_d2h,
               "total": t_h2d + t_kernel + t_d2h, "bloques": bloques, "hilos": hilos}
    return salida, tiempos


def nombre_gpu():
    nombre = cuda.get_current_device().name
    return nombre.decode() if isinstance(nombre, bytes) else nombre


def calentar():
    """Compila los kernels (JIT) con una imagen pequeña para no medir la compilación."""
    pequeña = generar_imagen(64, 64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")               # aviso de baja ocupación por el grid pequeño
        blur_cuda(pequeña, "global")
        blur_cuda(pequeña, "shared")


def main():
    if len(sys.argv) == 3:
        img = generar_imagen(int(sys.argv[1]), int(sys.argv[2]))
    else:
        img = obtener_imagen_8k()
    print(describir(img))
    print(f"GPU: {nombre_gpu()}")

    calentar()
    resultados = {}
    for kernel in ("global", "shared"):
        salida, t = blur_cuda(img, kernel)
        resultados[kernel] = salida
        print(f"\nKernel {kernel}: grid = {t['bloques'][0]}x{t['bloques'][1]} bloques "
              f"de {t['hilos'][0]}x{t['hilos'][1]} hilos")
        print(f"  H2D    : {t['h2d'] * 1000:9.3f} ms")
        print(f"  Kernel : {t['kernel'] * 1000:9.3f} ms")
        print(f"  D2H    : {t['d2h'] * 1000:9.3f} ms")
        print(f"  Total  : {t['total'] * 1000:9.3f} ms")
    print(f"\nAmbos kernels producen la misma imagen: {np.array_equal(resultados['global'], resultados['shared'])}")


if __name__ == "__main__":
    main()
