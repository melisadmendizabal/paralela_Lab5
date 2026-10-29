
import os

import numpy as np

ANCHO_8K = 7680
ALTO_8K = 4320
SEMILLA = 2026
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagen_8k.npy")


def _ruido_suave(alto, ancho, celda, rng):
    """Ruido de valor: rejilla aleatoria interpolada bilinealmente a (alto, ancho)."""
    gh, gw = alto // celda + 2, ancho // celda + 2
    rejilla = rng.random((gh, gw), dtype=np.float32)
    ys = np.arange(alto, dtype=np.float32) / celda
    xs = np.arange(ancho, dtype=np.float32) / celda
    y0, x0 = ys.astype(np.int32), xs.astype(np.int32)
    fy, fx = (ys - y0)[:, None], xs - x0
    fx = fx * fx * (3 - 2 * fx)                          # suavizado (smoothstep)
    fy = fy * fy * (3 - 2 * fy)
    filas = rejilla[:, x0] * (1 - fx) + rejilla[:, x0 + 1] * fx   # (gh, ancho)
    return filas[y0] * (1 - fy) + filas[y0 + 1] * fy               # (alto, ancho)


def generar_imagen(alto=ALTO_8K, ancho=ANCHO_8K, semilla=SEMILLA):
    """Devuelve una imagen RGB uint8 de forma (alto, ancho, 3) con aspecto de terreno."""
    rng = np.random.default_rng(semilla)
    altura = np.zeros((alto, ancho), dtype=np.float32)
    amplitud, total = 1.0, 0.0
    for celda in (1024, 512, 256, 128, 64, 32, 16, 8, 4):
        if celda < max(alto, ancho):
            altura += amplitud * _ruido_suave(alto, ancho, celda, rng)
            total += amplitud
        amplitud *= 0.55
    altura /= total

    paleta = np.array([[20, 50, 110], [40, 90, 150], [194, 178, 128], [70, 130, 50],
                       [35, 90, 35], [110, 95, 70], [150, 150, 150], [245, 245, 250]],
                      dtype=np.float32)
    cortes = np.array([0.0, 0.40, 0.46, 0.49, 0.58, 0.66, 0.74, 0.82], dtype=np.float32)
    img = np.empty((alto, ancho, 3), dtype=np.uint8)
    for c in range(3):
        canal = np.interp(altura, cortes, paleta[:, c]).astype(np.float32)
        canal += rng.normal(0.0, 12.0, (alto, ancho)).astype(np.float32)  # textura fina / ruido del sensor
        np.clip(canal, 0, 255, out=canal)
        img[:, :, c] = canal.astype(np.uint8)
    return img


def obtener_imagen_8k():
    """Imagen 8K generada una sola vez y guardada en imagen_8k.npy (tarda ~30 s en generarse)."""
    if os.path.exists(CACHE):
        return np.load(CACHE)
    img = generar_imagen()
    np.save(CACHE, img)
    return img


def cargar_imagen(ruta):
    """Carga una imagen real (PNG/JPG) como arreglo (alto, ancho, 3) uint8."""
    from PIL import Image
    return np.ascontiguousarray(np.asarray(Image.open(ruta).convert("RGB")))


def guardar_vista_general(img, ruta="imagen_8k_preview.png", factor=8):
    """Guarda la imagen reducida (1 de cada `factor` píxeles por eje) para mostrarla en el informe."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(ruta, img[::factor, ::factor])


def describir(img):
    alto, ancho, canales = img.shape
    mb = img.nbytes / (1024 ** 2)
    return (f"Imagen {ancho}x{alto} px, {canales} canales | Píxeles: {alto * ancho:,} "
            f"| Valores: {img.size:,} | dtype: {img.dtype} | Tamaño: {img.nbytes:,} bytes ({mb:.1f} MB)")


if __name__ == "__main__":
    imagen = obtener_imagen_8k()
    print(describir(imagen))
    guardar_vista_general(imagen)
    print("Vista general guardada en imagen_8k_preview.png")
