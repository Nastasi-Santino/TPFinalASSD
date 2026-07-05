"""
color.py

Módulo de color para extender el compresor DCT a imágenes RGB.

Implementa:
1. Conversión RGB -> YCbCr.
2. Conversión YCbCr -> RGB.
3. Separación de componentes Y, Cb, Cr.
4. Unión de componentes Y, Cb, Cr.
5. Conversión segura a uint8.
6. Guardado de imágenes RGB.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def rgb_to_ycbcr(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen RGB a YCbCr.

    Se usan las ecuaciones para valores digitales no signados de 8 bits:

        Y  =  0.299 R + 0.587 G + 0.114 B
        Cb = 128 - 0.168736 R - 0.331264 G + 0.5 B
        Cr = 128 + 0.5 R - 0.418688 G - 0.081312 B

    Parameters
    ----------
    image_rgb : np.ndarray
        Imagen RGB con forma (alto, ancho, 3).

    Returns
    -------
    image_ycbcr : np.ndarray
        Imagen YCbCr con forma (alto, ancho, 3), en float64.
        El orden de canales es (Y, Cb, Cr).
    """
    image_rgb = np.asarray(image_rgb, dtype=np.float64)

    _validate_color_image(image_rgb, name="image_rgb")

    r = image_rgb[:, :, 0]
    g = image_rgb[:, :, 1]
    b = image_rgb[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    image_ycbcr = np.stack((y, cb, cr), axis=2)

    return image_ycbcr


def ycbcr_to_rgb(image_ycbcr: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen YCbCr a RGB.

    Se usan las ecuaciones inversas:

        R = Y + 1.402 (Cr - 128)
        G = Y - 0.344136 (Cb - 128) - 0.714136 (Cr - 128)
        B = Y + 1.772 (Cb - 128)

    Parameters
    ----------
    image_ycbcr : np.ndarray
        Imagen YCbCr con forma (alto, ancho, 3).
        El orden esperado es (Y, Cb, Cr).

    Returns
    -------
    image_rgb : np.ndarray
        Imagen RGB reconstruida con forma (alto, ancho, 3), en float64.
    """
    image_ycbcr = np.asarray(image_ycbcr, dtype=np.float64)

    _validate_color_image(image_ycbcr, name="image_ycbcr")

    y = image_ycbcr[:, :, 0]
    cb = image_ycbcr[:, :, 1]
    cr = image_ycbcr[:, :, 2]

    cb_centered = cb - 128.0
    cr_centered = cr - 128.0

    r = y + 1.402 * cr_centered
    g = y - 0.344136 * cb_centered - 0.714136 * cr_centered
    b = y + 1.772 * cb_centered

    image_rgb = np.stack((r, g, b), axis=2)

    return image_rgb


def split_ycbcr_channels(image_ycbcr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separa una imagen YCbCr en sus tres componentes.

    Parameters
    ----------
    image_ycbcr : np.ndarray
        Imagen YCbCr con forma (alto, ancho, 3).

    Returns
    -------
    y : np.ndarray
        Componente de luminancia.

    cb : np.ndarray
        Componente de crominancia azul.

    cr : np.ndarray
        Componente de crominancia roja.
    """
    image_ycbcr = np.asarray(image_ycbcr, dtype=np.float64)

    _validate_color_image(image_ycbcr, name="image_ycbcr")

    y = image_ycbcr[:, :, 0]
    cb = image_ycbcr[:, :, 1]
    cr = image_ycbcr[:, :, 2]

    return y, cb, cr


def merge_ycbcr_channels(
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray
) -> np.ndarray:
    """
    Une tres componentes Y, Cb y Cr en una imagen YCbCr.

    Parameters
    ----------
    y : np.ndarray
        Componente de luminancia, forma (alto, ancho).

    cb : np.ndarray
        Componente Cb, forma (alto, ancho).

    cr : np.ndarray
        Componente Cr, forma (alto, ancho).

    Returns
    -------
    image_ycbcr : np.ndarray
        Imagen YCbCr con forma (alto, ancho, 3).
    """
    y = np.asarray(y, dtype=np.float64)
    cb = np.asarray(cb, dtype=np.float64)
    cr = np.asarray(cr, dtype=np.float64)

    _validate_channel(y, name="y")
    _validate_channel(cb, name="cb")
    _validate_channel(cr, name="cr")

    if y.shape != cb.shape or y.shape != cr.shape:
        raise ValueError(
            "Las componentes Y, Cb y Cr deben tener la misma forma. "
            f"Y={y.shape}, Cb={cb.shape}, Cr={cr.shape}."
        )

    return np.stack((y, cb, cr), axis=2)


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen float a uint8 con redondeo y saturación.

    Parameters
    ----------
    image : np.ndarray
        Imagen 2D o 3D.

    Returns
    -------
    image_uint8 : np.ndarray
        Imagen uint8 con valores en [0, 255].
    """
    image = np.asarray(image, dtype=np.float64)

    if image.ndim not in (2, 3):
        raise ValueError("La imagen debe tener 2 o 3 dimensiones.")

    image = np.clip(image, 0.0, 255.0)
    image = np.rint(image)

    return image.astype(np.uint8)


def save_rgb_image(image_rgb: np.ndarray, output_path: str | Path) -> None:
    """
    Guarda una imagen RGB en disco.

    Parameters
    ----------
    image_rgb : np.ndarray
        Imagen RGB con forma (alto, ancho, 3).

    output_path : str | Path
        Ruta de salida.
    """
    image_rgb = np.asarray(image_rgb)

    _validate_color_image(image_rgb, name="image_rgb")

    output_path = Path(output_path)

    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    image_uint8 = image_to_uint8(image_rgb)
    img = Image.fromarray(image_uint8, mode="RGB")
    img.save(output_path)


def _validate_color_image(image: np.ndarray, name: str = "image") -> None:
    """
    Valida que una imagen sea de color con tres canales.

    Parameters
    ----------
    image : np.ndarray
        Imagen a validar.

    name : str
        Nombre descriptivo para mensajes de error.
    """
    if image.ndim != 3:
        raise ValueError(f"{name} debe tener forma (alto, ancho, 3).")

    if image.shape[2] != 3:
        raise ValueError(
            f"{name} debe tener exactamente 3 canales. "
            f"Se recibió forma {image.shape}."
        )

    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"{name} debe tener dimensiones espaciales positivas.")


def _validate_channel(channel: np.ndarray, name: str = "channel") -> None:
    """
    Valida que una componente sea una imagen 2D.

    Parameters
    ----------
    channel : np.ndarray
        Canal a validar.

    name : str
        Nombre descriptivo para mensajes de error.
    """
    if channel.ndim != 2:
        raise ValueError(f"{name} debe ser una matriz 2D.")

    if channel.shape[0] <= 0 or channel.shape[1] <= 0:
        raise ValueError(f"{name} debe tener dimensiones positivas.")

from preprocessing import (
    pad_image_edge,
    split_into_blocks,
    level_shift_blocks,
    PreprocessingInfo,
)

from dct import dct2_blocks, idct2_blocks
from quantization import quantize_blocks, dequantize_blocks
from coding import encode_quantized_blocks, decode_quantized_blocks
from reconstruction import reconstruct_from_blocks


def preprocess_color_channel(channel: np.ndarray, block_size: int = 8) -> tuple[np.ndarray, PreprocessingInfo]:
    """
    Preprocesa una componente 2D de una imagen color.

    Realiza:
    1. padding por replicación de borde;
    2. división en bloques;
    3. corrimiento de nivel.

    Parameters
    ----------
    channel : np.ndarray
        Componente 2D, por ejemplo Y, Cb o Cr.

    block_size : int
        Tamaño de bloque.

    Returns
    -------
    shifted_blocks : np.ndarray
        Bloques con level shift aplicado.

    info : PreprocessingInfo
        Información necesaria para reconstruir la componente.
    """
    channel = np.asarray(channel, dtype=np.float64)

    _validate_channel(channel, name="channel")

    original_shape = channel.shape

    padded_channel, pad_height, pad_width = pad_image_edge(
        channel,
        block_size=block_size
    )

    padded_shape = padded_channel.shape

    blocks = split_into_blocks(
        padded_channel,
        block_size=block_size
    )

    shifted_blocks = level_shift_blocks(
        blocks,
        shift=128.0
    )

    info = PreprocessingInfo(
        original_shape=original_shape,
        padded_shape=padded_shape,
        pad_height=pad_height,
        pad_width=pad_width,
        block_size=block_size
    )

    return shifted_blocks, info


def compress_color_channel(
    channel: np.ndarray,
    q_matrix: np.ndarray,
    dct_matrix: np.ndarray,
    block_size: int = 8
) -> tuple[list, np.ndarray, PreprocessingInfo]:
    """
    Comprime una componente 2D de una imagen color.

    Realiza:
    1. preprocesamiento;
    2. DCT;
    3. cuantización;
    4. codificación zig-zag, DC diferencial y RLE.

    Parameters
    ----------
    channel : np.ndarray
        Componente 2D a comprimir.

    q_matrix : np.ndarray
        Matriz de cuantización correspondiente a la componente.

    dct_matrix : np.ndarray
        Matriz DCT.

    block_size : int
        Tamaño de bloque.

    Returns
    -------
    encoded_blocks : list
        Bloques codificados.

    quantized_blocks : np.ndarray
        Bloques cuantizados.

    info : PreprocessingInfo
        Información necesaria para reconstrucción.
    """
    shifted_blocks, info = preprocess_color_channel(
        channel,
        block_size=block_size
    )

    coeff_blocks = dct2_blocks(
        shifted_blocks,
        dct_matrix=dct_matrix
    )

    quantized_blocks = quantize_blocks(
        coeff_blocks,
        q_matrix=q_matrix
    )

    encoded_blocks = encode_quantized_blocks(
        quantized_blocks
    )

    return encoded_blocks, quantized_blocks, info


def reconstruct_color_channel(
    encoded_blocks: list,
    blocks_shape: tuple[int, int, int, int],
    q_matrix: np.ndarray,
    info: PreprocessingInfo,
    dct_matrix: np.ndarray
) -> np.ndarray:
    """
    Reconstruye una componente 2D de una imagen color.

    Realiza:
    1. decodificación;
    2. de-cuantización;
    3. IDCT;
    4. inverse level shift;
    5. reensamblado;
    6. crop.

    Parameters
    ----------
    encoded_blocks : list
        Bloques codificados.

    blocks_shape : tuple[int, int, int, int]
        Forma original del arreglo de bloques cuantizados.

    q_matrix : np.ndarray
        Matriz de cuantización usada para la componente.

    info : PreprocessingInfo
        Información de preprocesamiento.

    dct_matrix : np.ndarray
        Matriz DCT.

    Returns
    -------
    reconstructed_channel : np.ndarray
        Componente reconstruida, en uint8.
    """
    quantized_blocks = decode_quantized_blocks(
        encoded_blocks=encoded_blocks,
        blocks_shape=blocks_shape
    )

    dequantized_blocks = dequantize_blocks(
        quantized_blocks,
        q_matrix=q_matrix
    )

    reconstructed_shifted_blocks = idct2_blocks(
        dequantized_blocks,
        dct_matrix=dct_matrix
    )

    reconstructed_channel = reconstruct_from_blocks(
        reconstructed_shifted_blocks,
        info
    )

    return reconstructed_channel

if __name__ == "__main__":
    # Prueba simple de ida y vuelta RGB -> YCbCr -> RGB.
    from preprocessing import load_image

    image_path = "imagen_prueba_color.png"
    output_path = "color_roundtrip.png"

    rgb = load_image(image_path)
    ycbcr = rgb_to_ycbcr(rgb)
    rgb_reconstructed = ycbcr_to_rgb(ycbcr)

    error = np.max(np.abs(rgb.astype(np.float64) - rgb_reconstructed))

    print("Prueba de conversión de color")
    print("-----------------------------")
    print(f"Forma RGB original: {rgb.shape}")
    print(f"Forma YCbCr: {ycbcr.shape}")
    print(f"Error máximo RGB -> YCbCr -> RGB: {error:.6e}")

    save_rgb_image(rgb_reconstructed, output_path)
    print(f"Imagen guardada en: {output_path}")