
import json
import platform
import statistics
import time

import numpy as np
from numba import cuda

from blur_cuda import blur_cuda, calentar, nombre_gpu
from blur_secuencial import blur_numpy, blur_secuencial
from imagen import describir, obtener_imagen_8k

REPETICIONES = 5


def mediana_fases(corridas):
    """Mediana de cada fase; el total es la suma de esas medianas (T = H2D + kernel + D2H)."""
    med = {f: statistics.median(c[f] for c in corridas) for f in ("h2d", "kernel", "d2h")}
    med["total"] = med["h2d"] + med["kernel"] + med["d2h"]
    return med


def guardar_preview(img, salida, ruta="blur_preview.png"):
    """Recorte de 160x240 px ampliado x3: original (izquierda) vs desenfocada (derecha)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y, x = 2080, 2400                     # zona de costa (agua y tierra)
    a = img[y:y + 160, x:x + 240].repeat(3, 0).repeat(3, 1)
    b = salida[y:y + 160, x:x + 240].repeat(3, 0).repeat(3, 1)
    separador = np.full((a.shape[0], 12, 3), 255, dtype=np.uint8)
    plt.imsave(ruta, np.concatenate([a, separador, b], axis=1))


def main():
    img = obtener_imagen_8k()
    print(describir(img))
    print(f"CPU: {platform.processor()} | GPU: {nombre_gpu()}\n")

    print("Ejecutando versión secuencial en Python puro (puede tardar unos minutos)...")
    t0 = time.perf_counter()
    ref = blur_secuencial(img)
    t_cpu = time.perf_counter() - t0
    print(f"  T_cpu (Python puro) = {t_cpu:.3f} s")

    t0 = time.perf_counter()
    ref_np = blur_numpy(img)
    t_np = time.perf_counter() - t0
    assert np.array_equal(ref_np, ref), "NumPy: imagen distinta a la secuencial"
    print(f"  T_cpu (NumPy vectorizado, referencia adicional) = {t_np:.3f} s  ✔ imagen idéntica\n")

    calentar()
    img_pinned = cuda.pinned_array(img.shape, dtype=img.dtype)
    img_pinned[:] = img
    salida_pinned = cuda.pinned_array(img.shape, dtype=img.dtype)

    resultados = {"imagen": list(img.shape), "bytes": int(img.nbytes), "gpu": nombre_gpu(),
                  "t_cpu_s": t_cpu, "t_numpy_s": t_np, "repeticiones": REPETICIONES, "cuda": {}}

    for etiqueta, kernel, entrada, destino in (("global", "global", img, None),
                                               ("shared", "shared", img, None),
                                               ("shared_pinned", "shared", img_pinned, salida_pinned)):
        corridas = []
        for _ in range(REPETICIONES):
            salida, t = blur_cuda(entrada, kernel, destino)
            assert np.array_equal(salida, ref), f"{etiqueta}: imagen distinta a la secuencial"
            corridas.append(t)
        med = mediana_fases(corridas)
        speedup = t_cpu / med["total"]
        resultados["cuda"][etiqueta] = {**med, "bloques": t["bloques"], "hilos": t["hilos"],
                                        "speedup_total": speedup,
                                        "speedup_solo_kernel": t_cpu / med["kernel"],
                                        "speedup_vs_numpy": t_np / med["total"],
                                        "ancho_banda_h2d_GBs": img.nbytes / med["h2d"] / 1e9,
                                        "ancho_banda_d2h_GBs": img.nbytes / med["d2h"] / 1e9}
        print(f"[{etiqueta}] grid = {t['bloques'][0]}x{t['bloques'][1]} bloques de "
              f"{t['hilos'][0]}x{t['hilos'][1]} hilos (mediana de {REPETICIONES})")
        print(f"  H2D {med['h2d'] * 1e3:7.2f} ms ({img.nbytes / med['h2d'] / 1e9:.2f} GB/s) | "
              f"Kernel {med['kernel'] * 1e3:6.2f} ms | "
              f"D2H {med['d2h'] * 1e3:7.2f} ms ({img.nbytes / med['d2h'] / 1e9:.2f} GB/s) | "
              f"Total {med['total'] * 1e3:7.2f} ms")
        print(f"  Speedup = {t_cpu:.3f} s / {med['total']:.4f} s = {speedup:,.1f}x "
              f"(solo kernel: {t_cpu / med['kernel']:,.0f}x | vs NumPy: {t_np / med['total']:.1f}x)"
              f"  ✔ imagen idéntica\n")

    guardar_preview(img, ref)
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print("Resultados guardados en resultados.json y vista previa en blur_preview.png")


if __name__ == "__main__":
    main()
