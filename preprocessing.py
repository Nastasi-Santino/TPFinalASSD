"""
preprocessing.py

Módulo de preprocesamiento para un compresor de imágenes basado en DCT.

Etapas implementadas:
1. Carga de imagen.
2. Conversión a escala de grises.
3. Relleno hasta dimensiones múltiplos de 8 usando replicación de borde.
4. División en bloques no solapados de 8x8.
5. Corrimiento de nivel restando 128.

No se utiliza ninguna librería que implemente directamente el algoritmo de compresión.
Solo se usan librerías auxiliares para manejo de arreglos e imágenes.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


BLOCK_SIZE = 8


@dataclass
class PreprocessingInfo:
    """
    Información necesaria para reconstruir correctamente la imagen más adelante.

    original_shape:
        Dimensiones originales de la imagen en escala de grises, antes del padding.
        Formato: (alto, ancho)

    padded_shape:
        Dimensiones de la imagen luego del padding.
        Formato: (alto_padded, ancho_padded)

    pad_height:
        Cantidad de filas agregadas al final.

    pad_width:
        Cantidad de columnas agregadas al final.

    block_size:
        Tamaño de bloque utilizado.
    """
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    pad_height: int
    pad_width: int
    block_size: int = BLOCK_SIZE


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Carga una imagen desde disco y la devuelve como arreglo NumPy.

    La imagen se carga en modo RGB para tener una representación consistente,
    incluso si el archivo original estaba en escala de grises, RGBA, etc.

    Parameters
    ----------
    image_path : str | Path
        Ruta al archivo de imagen.

    Returns
    -------
    image_rgb : np.ndarray
        Imagen RGB con forma (alto, ancho, 3) y tipo uint8.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    with Image.open(image_path) as img:
        image_rgb = img.convert("RGB")
        image_rgb = np.array(image_rgb, dtype=np.uint8)

    return image_rgb


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen RGB a escala de grises.

    Se utiliza la combinación ponderada habitual para luminancia:

        Y = 0.299 R + 0.587 G + 0.114 B

    Esta conversión aproxima la sensibilidad del ojo humano,
    que es mayor para el verde, intermedia para el rojo y menor para el azul.

    Parameters
    ----------
    image : np.ndarray
        Imagen de entrada. Puede tener forma:
        - (alto, ancho): imagen ya en escala de grises.
        - (alto, ancho, 3): imagen RGB.
        - (alto, ancho, 4): imagen RGBA, se ignora el canal alfa.

    Returns
    -------
    gray : np.ndarray
        Imagen en escala de grises con forma (alto, ancho) y tipo float32.
    """
    if image.ndim == 2:
        return image.astype(np.float32)

    if image.ndim != 3:
        raise ValueError(
            "La imagen debe tener forma (alto, ancho) o (alto, ancho, canales)."
        )

    if image.shape[2] < 3:
        raise ValueError(
            "La imagen debe tener al menos 3 canales para convertir desde RGB."
        )

    # Se toman solamente los canales RGB. Si hay canal alfa, se ignora.
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    return gray.astype(np.float32)


def compute_padding(height: int, width: int, block_size: int = BLOCK_SIZE) -> tuple[int, int]:
    """
    Calcula cuántas filas y columnas deben agregarse para que
    la imagen tenga dimensiones múltiplos de block_size.

    Parameters
    ----------
    height : int
        Alto original de la imagen.

    width : int
        Ancho original de la imagen.

    block_size : int
        Tamaño de bloque deseado.

    Returns
    -------
    pad_height : int
        Cantidad de filas a agregar.

    pad_width : int
        Cantidad de columnas a agregar.
    """
    if height <= 0 or width <= 0:
        raise ValueError("Las dimensiones de la imagen deben ser positivas.")

    if block_size <= 0:
        raise ValueError("El tamaño de bloque debe ser positivo.")

    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size

    return pad_height, pad_width


def pad_image_edge(image: np.ndarray, block_size: int = BLOCK_SIZE) -> tuple[np.ndarray, int, int]:
    """
    Rellena la imagen hasta que sus dimensiones sean múltiplos de block_size.

    El relleno se realiza por replicación de borde. Es decir, las filas
    agregadas replican la última fila válida y las columnas agregadas replican
    la última columna válida.

    Parameters
    ----------
    image : np.ndarray
        Imagen en escala de grises con forma (alto, ancho).

    block_size : int
        Tamaño de bloque deseado.

    Returns
    -------
    padded_image : np.ndarray
        Imagen con dimensiones múltiplos de block_size.

    pad_height : int
        Cantidad de filas agregadas.

    pad_width : int
        Cantidad de columnas agregadas.
    """
    if image.ndim != 2:
        raise ValueError("El padding espera una imagen 2D en escala de grises.")

    height, width = image.shape
    pad_height, pad_width = compute_padding(height, width, block_size)

    padded_image = np.pad(
        image,
        pad_width=((0, pad_height), (0, pad_width)),
        mode="edge"
    )

    return padded_image.astype(np.float32), pad_height, pad_width


def split_into_blocks(image: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """
    Divide una imagen 2D en bloques no solapados de block_size x block_size.

    La imagen debe tener dimensiones múltiplos de block_size.

    Parameters
    ----------
    image : np.ndarray
        Imagen en escala de grises, ya con padding aplicado.
        Forma: (alto_padded, ancho_padded)

    block_size : int
        Tamaño de bloque.

    Returns
    -------
    blocks : np.ndarray
        Arreglo de bloques con forma:

            (num_bloques_vertical, num_bloques_horizontal, block_size, block_size)

        Por ejemplo, para una imagen de 256x256 y bloques de 8x8:
            blocks.shape = (32, 32, 8, 8)
    """
    if image.ndim != 2:
        raise ValueError("La imagen debe ser 2D para dividirla en bloques.")

    height, width = image.shape

    if height % block_size != 0 or width % block_size != 0:
        raise ValueError(
            "Las dimensiones de la imagen deben ser múltiplos del tamaño de bloque."
        )

    num_blocks_h = height // block_size
    num_blocks_w = width // block_size

    blocks = image.reshape(
        num_blocks_h,
        block_size,
        num_blocks_w,
        block_size
    )

    blocks = blocks.transpose(0, 2, 1, 3)

    return blocks.astype(np.float32)


def level_shift_blocks(blocks: np.ndarray, shift: float = 128.0) -> np.ndarray:
    """
    Aplica corrimiento de nivel a los bloques.

    Para imágenes de 8 bits, se resta 128 para pasar aproximadamente de:

        [0, 255]  a  [-128, 127]

    Parameters
    ----------
    blocks : np.ndarray
        Bloques de imagen con forma:
        (num_bloques_vertical, num_bloques_horizontal, block_size, block_size)

    shift : float
        Valor a restar. Para imágenes de 8 bits se usa 128.

    Returns
    -------
    shifted_blocks : np.ndarray
        Bloques con corrimiento de nivel aplicado.
    """
    if blocks.ndim != 4:
        raise ValueError(
            "Los bloques deben tener forma "
            "(num_bloques_vertical, num_bloques_horizontal, block_size, block_size)."
        )

    shifted_blocks = blocks.astype(np.float32) - shift

    return shifted_blocks


def preprocess_image(image_path: str | Path, block_size: int = BLOCK_SIZE) -> tuple[np.ndarray, PreprocessingInfo]:
    """
    Ejecuta el preprocesamiento completo de una imagen.

    Etapas:
    1. Carga de imagen.
    2. Conversión a escala de grises.
    3. Padding por replicación de borde.
    4. División en bloques de 8x8.
    5. Corrimiento de nivel.

    Parameters
    ----------
    image_path : str | Path
        Ruta a la imagen de entrada.

    block_size : int
        Tamaño de bloque. Por defecto, 8.

    Returns
    -------
    shifted_blocks : np.ndarray
        Bloques preprocesados y centrados alrededor de cero.
        Forma:
        (num_bloques_vertical, num_bloques_horizontal, block_size, block_size)

    info : PreprocessingInfo
        Información necesaria para la etapa de reconstrucción.
    """
    image_rgb = load_image(image_path)

    gray = rgb_to_grayscale(image_rgb)
    original_shape = gray.shape

    padded_gray, pad_height, pad_width = pad_image_edge(gray, block_size)
    padded_shape = padded_gray.shape

    blocks = split_into_blocks(padded_gray, block_size)
    shifted_blocks = level_shift_blocks(blocks, shift=128.0)

    info = PreprocessingInfo(
        original_shape=original_shape,
        padded_shape=padded_shape,
        pad_height=pad_height,
        pad_width=pad_width,
        block_size=block_size
    )

    return shifted_blocks, info


if __name__ == "__main__":
    # Ejemplo mínimo de uso.
    # Cambiar esta ruta por la imagen que quieran probar.
    image_path = "imagen_prueba.png"

    blocks, info = preprocess_image(image_path)

    print("Preprocesamiento completado.")
    print(f"Dimensiones originales: {info.original_shape}")
    print(f"Dimensiones con padding: {info.padded_shape}")
    print(f"Padding agregado: filas={info.pad_height}, columnas={info.pad_width}")
    print(f"Forma del arreglo de bloques: {blocks.shape}")
    print(f"Rango de valores luego del level shift: [{blocks.min():.2f}, {blocks.max():.2f}]")