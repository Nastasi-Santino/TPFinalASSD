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
from dataclasses import dataclass

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

@dataclass
class ChromaSubsamplingInfo:
    """
    Información necesaria para invertir el submuestreo de crominancia.

    original_shape:
        Forma original de las componentes Cb y Cr antes del padding.

    padded_shape:
        Forma luego de agregar padding para que las dimensiones sean pares.

    pad_height:
        Cantidad de filas agregadas.

    pad_width:
        Cantidad de columnas agregadas.

    subsampled_shape:
        Forma de la componente luego del submuestreo 4:2:0.

    mode:
        Modo de submuestreo utilizado.
    """
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    pad_height: int
    pad_width: int
    subsampled_shape: tuple[int, int]
    mode: str = "4:2:0"


def pad_channel_to_even_shape(channel: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Agrega padding por replicación de borde para que una componente tenga
    dimensiones pares.

    Esto es necesario para aplicar submuestreo 4:2:0 mediante bloques 2x2.

    Parameters
    ----------
    channel : np.ndarray
        Componente 2D a completar.

    Returns
    -------
    padded_channel : np.ndarray
        Componente con dimensiones pares.

    pad_height : int
        Cantidad de filas agregadas.

    pad_width : int
        Cantidad de columnas agregadas.
    """
    channel = np.asarray(channel, dtype=np.float64)

    _validate_channel(channel, name="channel")

    height, width = channel.shape

    pad_height = height % 2
    pad_width = width % 2

    padded_channel = np.pad(
        channel,
        pad_width=((0, pad_height), (0, pad_width)),
        mode="edge"
    )

    return padded_channel.astype(np.float64), pad_height, pad_width


def downsample_420_channel(channel: np.ndarray) -> tuple[np.ndarray, ChromaSubsamplingInfo]:
    """
    Submuestrea una componente de crominancia usando esquema 4:2:0.

    El procedimiento implementado es:

    1. Padding por replicación de borde hasta obtener dimensiones pares.
    2. Promedio local 2x2.
    3. Decimación por 2 en ambas dimensiones.

    Matemáticamente, para una componente c[m,n]:

        c_d[p,q] = 1/4 * (
            c[2p,   2q  ] +
            c[2p+1, 2q  ] +
            c[2p,   2q+1] +
            c[2p+1, 2q+1]
        )

    Esta operación puede interpretarse como un filtro pasabajos 2x2 seguido
    de downsampling por 2 en vertical y horizontal.

    Parameters
    ----------
    channel : np.ndarray
        Componente 2D de crominancia, por ejemplo Cb o Cr.

    Returns
    -------
    downsampled_channel : np.ndarray
        Componente submuestreada.

    info : ChromaSubsamplingInfo
        Información necesaria para reconstruir la forma original.
    """
    channel = np.asarray(channel, dtype=np.float64)

    _validate_channel(channel, name="channel")

    original_shape = channel.shape

    padded_channel, pad_height, pad_width = pad_channel_to_even_shape(channel)
    padded_shape = padded_channel.shape

    downsampled_channel = (
        padded_channel[0::2, 0::2]
        + padded_channel[1::2, 0::2]
        + padded_channel[0::2, 1::2]
        + padded_channel[1::2, 1::2]
    ) / 4.0

    info = ChromaSubsamplingInfo(
        original_shape=original_shape,
        padded_shape=padded_shape,
        pad_height=pad_height,
        pad_width=pad_width,
        subsampled_shape=downsampled_channel.shape,
        mode="4:2:0"
    )

    return downsampled_channel.astype(np.float64), info


def downsample_chrominance_420(
    cb: np.ndarray,
    cr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, ChromaSubsamplingInfo]:
    """
    Aplica submuestreo 4:2:0 a las componentes Cb y Cr.

    Ambas componentes deben tener la misma forma. Se aplica el mismo criterio
    de padding y submuestreo a las dos.

    Parameters
    ----------
    cb : np.ndarray
        Componente Cb.

    cr : np.ndarray
        Componente Cr.

    Returns
    -------
    cb_down : np.ndarray
        Componente Cb submuestreada.

    cr_down : np.ndarray
        Componente Cr submuestreada.

    info : ChromaSubsamplingInfo
        Información de submuestreo común a ambas componentes.
    """
    cb = np.asarray(cb, dtype=np.float64)
    cr = np.asarray(cr, dtype=np.float64)

    _validate_channel(cb, name="cb")
    _validate_channel(cr, name="cr")

    if cb.shape != cr.shape:
        raise ValueError(
            "Cb y Cr deben tener la misma forma para aplicar 4:2:0. "
            f"Cb={cb.shape}, Cr={cr.shape}."
        )

    cb_down, info_cb = downsample_420_channel(cb)
    cr_down, info_cr = downsample_420_channel(cr)

    _validate_same_subsampling_info(info_cb, info_cr)

    return cb_down, cr_down, info_cb


def resize_channel_bilinear(channel: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Redimensiona una componente 2D usando interpolación bilineal.

    Esta función se usa para reconstruir Cb y Cr desde la resolución
    submuestreada hacia la resolución original.

    Parameters
    ----------
    channel : np.ndarray
        Componente 2D de entrada.

    target_shape : tuple[int, int]
        Forma deseada:
        (alto_objetivo, ancho_objetivo)

    Returns
    -------
    resized_channel : np.ndarray
        Componente redimensionada mediante interpolación bilineal.
    """
    channel = np.asarray(channel, dtype=np.float64)

    _validate_channel(channel, name="channel")
    _validate_shape_tuple(target_shape, name="target_shape")

    src_height, src_width = channel.shape
    target_height, target_width = target_shape

    if src_height == target_height and src_width == target_width:
        return channel.copy()

    # Mapeo alineado a centros de píxel.
    # Esto evita asumir que las muestras están ubicadas exactamente en los bordes.
    row_scale = src_height / target_height
    col_scale = src_width / target_width

    target_rows = np.arange(target_height)
    target_cols = np.arange(target_width)

    src_rows = (target_rows + 0.5) * row_scale - 0.5
    src_cols = (target_cols + 0.5) * col_scale - 0.5

    src_rows = np.clip(src_rows, 0.0, src_height - 1.0)
    src_cols = np.clip(src_cols, 0.0, src_width - 1.0)

    row0 = np.floor(src_rows).astype(int)
    col0 = np.floor(src_cols).astype(int)

    row1 = np.clip(row0 + 1, 0, src_height - 1)
    col1 = np.clip(col0 + 1, 0, src_width - 1)

    row_weight = src_rows - row0
    col_weight = src_cols - col0

    resized_channel = np.zeros((target_height, target_width), dtype=np.float64)

    for i in range(target_height):
        r0 = row0[i]
        r1 = row1[i]
        wr = row_weight[i]

        top = (
            (1.0 - col_weight) * channel[r0, col0]
            + col_weight * channel[r0, col1]
        )

        bottom = (
            (1.0 - col_weight) * channel[r1, col0]
            + col_weight * channel[r1, col1]
        )

        resized_channel[i, :] = (1.0 - wr) * top + wr * bottom

    return resized_channel


def upsample_420_channel_bilinear(
    downsampled_channel: np.ndarray,
    info: ChromaSubsamplingInfo
) -> np.ndarray:
    """
    Reconstruye una componente submuestreada 4:2:0 usando interpolación bilineal.

    El procedimiento es:

    1. Interpolar desde la forma submuestreada hasta la forma con padding.
    2. Recortar el padding para recuperar la forma original.

    Parameters
    ----------
    downsampled_channel : np.ndarray
        Componente submuestreada.

    info : ChromaSubsamplingInfo
        Información generada durante el submuestreo.

    Returns
    -------
    reconstructed_channel : np.ndarray
        Componente reconstruida a la forma original.
    """
    downsampled_channel = np.asarray(downsampled_channel, dtype=np.float64)

    _validate_channel(downsampled_channel, name="downsampled_channel")
    _validate_chroma_subsampling_info(info)

    if downsampled_channel.shape != info.subsampled_shape:
        raise ValueError(
            "La forma de downsampled_channel no coincide con la registrada "
            "en ChromaSubsamplingInfo. "
            f"Se recibió {downsampled_channel.shape}, "
            f"pero se esperaba {info.subsampled_shape}."
        )

    padded_reconstruction = resize_channel_bilinear(
        downsampled_channel,
        target_shape=info.padded_shape
    )

    original_height, original_width = info.original_shape

    reconstructed_channel = padded_reconstruction[
        :original_height,
        :original_width
    ]

    return reconstructed_channel.astype(np.float64)


def upsample_chrominance_420(
    cb_down: np.ndarray,
    cr_down: np.ndarray,
    info: ChromaSubsamplingInfo
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruye las componentes Cb y Cr desde submuestreo 4:2:0.

    Parameters
    ----------
    cb_down : np.ndarray
        Componente Cb submuestreada.

    cr_down : np.ndarray
        Componente Cr submuestreada.

    info : ChromaSubsamplingInfo
        Información de submuestreo.

    Returns
    -------
    cb_up : np.ndarray
        Componente Cb reconstruida a resolución completa.

    cr_up : np.ndarray
        Componente Cr reconstruida a resolución completa.
    """
    cb_down = np.asarray(cb_down, dtype=np.float64)
    cr_down = np.asarray(cr_down, dtype=np.float64)

    _validate_channel(cb_down, name="cb_down")
    _validate_channel(cr_down, name="cr_down")

    if cb_down.shape != cr_down.shape:
        raise ValueError(
            "Cb_down y Cr_down deben tener la misma forma. "
            f"Cb_down={cb_down.shape}, Cr_down={cr_down.shape}."
        )

    cb_up = upsample_420_channel_bilinear(cb_down, info)
    cr_up = upsample_420_channel_bilinear(cr_down, info)

    return cb_up, cr_up


def _validate_shape_tuple(shape: tuple[int, int], name: str = "shape") -> None:
    """
    Valida una tupla de forma 2D.

    Parameters
    ----------
    shape : tuple[int, int]
        Tupla esperada:
        (alto, ancho)

    name : str
        Nombre descriptivo para mensajes de error.
    """
    if not isinstance(shape, tuple):
        raise TypeError(f"{name} debe ser una tupla.")

    if len(shape) != 2:
        raise ValueError(f"{name} debe tener longitud 2: (alto, ancho).")

    height, width = shape

    if height <= 0 or width <= 0:
        raise ValueError(f"Las dimensiones de {name} deben ser positivas.")


def _validate_chroma_subsampling_info(info: ChromaSubsamplingInfo) -> None:
    """
    Valida un objeto ChromaSubsamplingInfo.

    Parameters
    ----------
    info : ChromaSubsamplingInfo
        Información de submuestreo.
    """
    if not isinstance(info, ChromaSubsamplingInfo):
        raise TypeError("info debe ser una instancia de ChromaSubsamplingInfo.")

    _validate_shape_tuple(info.original_shape, name="info.original_shape")
    _validate_shape_tuple(info.padded_shape, name="info.padded_shape")
    _validate_shape_tuple(info.subsampled_shape, name="info.subsampled_shape")

    original_height, original_width = info.original_shape
    padded_height, padded_width = info.padded_shape
    subsampled_height, subsampled_width = info.subsampled_shape

    if padded_height < original_height or padded_width < original_width:
        raise ValueError(
            "La forma con padding no puede ser menor que la forma original."
        )

    if info.pad_height < 0 or info.pad_width < 0:
        raise ValueError("Los valores de padding no pueden ser negativos.")

    if padded_height % 2 != 0 or padded_width % 2 != 0:
        raise ValueError(
            "La forma con padding debe tener dimensiones pares para 4:2:0."
        )

    expected_subsampled_shape = (padded_height // 2, padded_width // 2)

    if info.subsampled_shape != expected_subsampled_shape:
        raise ValueError(
            "La forma submuestreada no es consistente con la forma con padding. "
            f"Se esperaba {expected_subsampled_shape}, "
            f"pero se recibió {info.subsampled_shape}."
        )

    if subsampled_height <= 0 or subsampled_width <= 0:
        raise ValueError("La forma submuestreada debe tener dimensiones positivas.")

    if info.mode != "4:2:0":
        raise ValueError(
            f"Modo de submuestreo no soportado: {info.mode}. "
            "Actualmente solo se soporta '4:2:0'."
        )


def _validate_same_subsampling_info(
    info_a: ChromaSubsamplingInfo,
    info_b: ChromaSubsamplingInfo
) -> None:
    """
    Verifica que dos objetos ChromaSubsamplingInfo sean compatibles.

    Se usa para asegurar que Cb y Cr fueron submuestreadas de la misma forma.
    """
    _validate_chroma_subsampling_info(info_a)
    _validate_chroma_subsampling_info(info_b)

    if info_a != info_b:
        raise ValueError(
            "Las componentes Cb y Cr no tienen información de submuestreo compatible."
        )

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