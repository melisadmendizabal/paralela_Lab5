
import statistics
import time

import numpy as np
from numba import cuda, int32, uint8

from blur_cuda import (BLOQUE_X, BLOQUE_Y, CANALES, TILE_X, TILE_Y, blur_global, blur_shared,
                       configurar_grid)
from imagen import obtener_imagen_8k

REPETICIONES = 10


@cuda.jit
def blur_shared_v1(img, salida):
    tile = cuda.shared.array((TILE_Y, TILE_X, CANALES), dtype=uint8)
    alto, ancho = img.shape[0], img.shape[1]
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    origen_x = cuda.blockIdx.x * BLOQUE_X - 1
    origen_y = cuda.blockIdx.y * BLOQUE_Y - 1
    for k in range(ty * BLOQUE_X + tx, TILE_X * TILE_Y, BLOQUE_X * BLOQUE_Y):
        sy, sx = k // TILE_X, k % TILE_X
        gy, gx = origen_y + sy, origen_x + sx
        if gy >= 0 and gy < alto and gx >= 0 and gx < ancho:
            for c in range(CANALES):
                tile[sy, sx, c] = img[gy, gx, c]
    cuda.syncthreads()
    x, y = cuda.grid(2)
    if x < ancho and y < alto:
        y0, y1 = max(0, y - 1), min(alto - 1, y + 1)
        x0, x1 = max(0, x - 1), min(ancho - 1, x + 1)
        cuenta = int32((y1 - y0 + 1) * (x1 - x0 + 1))
        for c in range(CANALES):
            suma = int32(0)
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    suma += int32(tile[yy - origen_y, xx - origen_x, c])
            salida[y, x, c] = uint8((suma + cuenta // 2) // cuenta)


def main():
    img = obtener_imagen_8k()
    d_img = cuda.to_device(img)
    d_salida = cuda.device_array_like(img)
    bloques, hilos = configurar_grid(img.shape[0], img.shape[1])

    resultados = {}
    for nombre, kernel in (("blur_global", blur_global),
                           ("blur_shared_v1", blur_shared_v1),
                           ("blur_shared", blur_shared)):
        kernel[bloques, hilos](d_img, d_salida)          # calentamiento (compilación JIT)
        cuda.synchronize()
        tiempos = []
        for _ in range(REPETICIONES):
            t0 = time.perf_counter()
            kernel[bloques, hilos](d_img, d_salida)
            cuda.synchronize()
            tiempos.append(time.perf_counter() - t0)
        resultados[nombre] = d_salida.copy_to_host()
        print(f"{nombre:15s}: {statistics.median(tiempos) * 1e3:6.2f} ms (mediana de {REPETICIONES})")

    iguales = all(np.array_equal(resultados["blur_global"], r) for r in resultados.values())
    print(f"Las tres versiones producen la misma imagen: {iguales}")


if __name__ == "__main__":
    main()
