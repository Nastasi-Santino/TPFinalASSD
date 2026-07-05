"""
reconstruction.py

Módulo de reconstrucción para un compresor de imágenes basado en DCT.

Este módulo implementa la etapa final del decodificador/reconstructor:

1. Decodificación de bloques cuantizados a partir de símbolos codificados.
2. De-cuantización de coeficientes.
3. Transformada inversa de la DCT.
4. Corrimiento de nivel inverso.
5. Reensamblado de bloques.
6. Recorte al tamaño original.
7. Saturación al rango [0, 255].
8. Conversión a imagen uint8.
9. Guardado opcional de la imagen reconstruida.

Las operaciones de decodificación, de-cuantización e IDCT se realizan usando
funciones ya implementadas en los módulos coding.py, quantization.py y dct.py.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from preprocessing import PreprocessingInfo
from dct import create_dct_matrix, idct2_blocks
from quantization import dequantize_blocks
from coding import EncodedBlock, decode_quantized_blocks


BLOCK_SIZE = 8


def inverse_level_shift_blocks(blocks: np.ndarray, shift: float = 128.0) -> np.ndarray:
    """
    Deshace el corrimiento de nivel aplicado durante el preprocesamiento.

    Durante el preprocesamiento se restó 128 a cada píxel para centrar
    los valores alrededor de cero. En la reconstrucción se suma nuevamente
    ese valor.

    Parameters
    ----------
    blocks : np.ndarray
        Bloques reconstruidos luego de aplicar IDCT.
        Forma:
        (num_bloques_vertical, num_bloques_horizontal, N, N)

    shift : float
        Valor a sumar. Para imágenes de 8 bits se usa 128.

    Returns
    -------
    shifted_blocks : np.ndarray
        Bloques con el corrimiento de nivel inverso aplicado.
    """
    blocks = np.asarray(blocks, dtype=np.float64)

    _validate_blocks_array(blocks, name="blocks")

    return blocks + shift


def merge_blocks(blocks: np.ndarray) -> np.ndarray:
    """
    Reensambla una imagen 2D a partir de bloques no solapados.

    La función invierte la operación realizada en split_into_blocks()
    del módulo preprocessing.py.

    Si blocks tiene forma:

        (num_bloques_vertical, num_bloques_horizontal, N, N)

    la imagen reconstruida tendrá forma:

        (num_bloques_vertical * N, num_bloques_horizontal * N)

    Parameters
    ----------
    blocks : np.ndarray
        Arreglo de bloques con forma:
        (num_bloques_vertical, num_bloques_horizontal, N, N)

    Returns
    -------
    image : np.ndarray
        Imagen reensamblada en escala de grises.
    """
    blocks = np.asarray(blocks, dtype=np.float64)

    _validate_blocks_array(blocks, name="blocks")

    num_blocks_h, num_blocks_w, block_h, block_w = blocks.shape

    # Inverso de:
    # image.reshape(num_blocks_h, block_size, num_blocks_w, block_size)
    #      .transpose(0, 2, 1, 3)
    image = blocks.transpose(0, 2, 1, 3).reshape(
        num_blocks_h * block_h,
        num_blocks_w * block_w
    )

    return image


def crop_to_original_shape(image: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    """
    Recorta una imagen reconstruida a sus dimensiones originales.

    Esto elimina las filas y columnas agregadas durante el padding.

    Parameters
    ----------
    image : np.ndarray
        Imagen reconstruida con padding.

    original_shape : tuple[int, int]
        Forma original de la imagen antes del padding:
        (alto_original, ancho_original)

    Returns
    -------
    cropped_image : np.ndarray
        Imagen recortada a las dimensiones originales.
    """
    image = np.asarray(image, dtype=np.float64)

    if image.ndim != 2:
        raise ValueError("La imagen debe ser 2D.")

    _validate_image_shape_tuple(original_shape, name="original_shape")

    original_height, original_width = original_shape
    height, width = image.shape

    if original_height > height or original_width > width:
        raise ValueError(
            "original_shape no puede ser mayor que la forma de la imagen reconstruida. "
            f"Imagen reconstruida: {image.shape}, original_shape: {original_shape}."
        )

    return image[:original_height, :original_width]


def clip_image(image: np.ndarray, min_value: float = 0.0, max_value: float = 255.0) -> np.ndarray:
    """
    Satura los valores de una imagen a un rango válido.

    Para imágenes de 8 bits, el rango válido es [0, 255].

    Parameters
    ----------
    image : np.ndarray
        Imagen reconstruida.

    min_value : float
        Valor mínimo permitido.

    max_value : float
        Valor máximo permitido.

    Returns
    -------
    clipped_image : np.ndarray
        Imagen saturada al rango indicado.
    """
    image = np.asarray(image, dtype=np.float64)

    if image.ndim != 2:
        raise ValueError("La imagen debe ser 2D.")

    if max_value < min_value:
        raise ValueError("max_value debe ser mayor o igual que min_value.")

    return np.clip(image, min_value, max_value)


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen reconstruida a formato uint8.

    Antes de convertir, redondea al entero más cercano y satura al intervalo
    [0, 255].

    Parameters
    ----------
    image : np.ndarray
        Imagen reconstruida en formato float.

    Returns
    -------
    image_uint8 : np.ndarray
        Imagen lista para guardar o visualizar, con tipo uint8.
    """
    image = np.asarray(image, dtype=np.float64)

    if image.ndim != 2:
        raise ValueError("La imagen debe ser 2D.")

    image = clip_image(image, min_value=0.0, max_value=255.0)
    image = np.rint(image)

    return image.astype(np.uint8)


def save_grayscale_image(image: np.ndarray, output_path: str | Path) -> None:
    """
    Guarda una imagen en escala de grises en disco.

    Parameters
    ----------
    image : np.ndarray
        Imagen 2D. Puede estar en float o uint8.

    output_path : str | Path
        Ruta donde se guardará la imagen.
    """
    output_path = Path(output_path)

    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    image_uint8 = image_to_uint8(image)

    img = Image.fromarray(image_uint8, mode="L")
    img.save(output_path)


def reconstruct_from_blocks(
    reconstructed_shifted_blocks: np.ndarray,
    info: PreprocessingInfo
) -> np.ndarray:
    """
    Reconstruye una imagen final a partir de bloques espaciales con level shift.

    Esta función asume que los bloques ya pasaron por:
    - de-cuantización;
    - IDCT.

    Por lo tanto, solo realiza:
    1. corrimiento de nivel inverso;
    2. reensamblado de bloques;
    3. crop al tamaño original;
    4. clipping;
    5. conversión a uint8.

    Parameters
    ----------
    reconstructed_shifted_blocks : np.ndarray
        Bloques reconstruidos todavía centrados alrededor de cero.
        Forma:
        (num_bloques_vertical, num_bloques_horizontal, N, N)

    info : PreprocessingInfo
        Información generada durante el preprocesamiento.

    Returns
    -------
    reconstructed_image : np.ndarray
        Imagen reconstruida en escala de grises, tipo uint8.
    """
    _validate_preprocessing_info(info)

    reconstructed_shifted_blocks = np.asarray(
        reconstructed_shifted_blocks,
        dtype=np.float64
    )

    _validate_blocks_array(
        reconstructed_shifted_blocks,
        name="reconstructed_shifted_blocks"
    )

    blocks = inverse_level_shift_blocks(reconstructed_shifted_blocks, shift=128.0)
    padded_image = merge_blocks(blocks)

    _validate_padded_shape(padded_image.shape, info.padded_shape)

    cropped_image = crop_to_original_shape(padded_image, info.original_shape)
    reconstructed_image = image_to_uint8(cropped_image)

    return reconstructed_image


def reconstruct_from_quantized_blocks(
    quantized_blocks: np.ndarray,
    q_matrix: np.ndarray,
    info: PreprocessingInfo,
    dct_matrix: np.ndarray | None = None
) -> np.ndarray:
    """
    Reconstruye una imagen a partir de bloques cuantizados.

    Esta función realiza:
    1. de-cuantización;
    2. IDCT;
    3. corrimiento de nivel inverso;
    4. reensamblado;
    5. crop;
    6. clipping y conversión a uint8.

    Parameters
    ----------
    quantized_blocks : np.ndarray
        Bloques de coeficientes cuantizados.
        Forma:
        (num_bloques_vertical, num_bloques_horizontal, N, N)

    q_matrix : np.ndarray
        Matriz de cuantización usada en la compresión.

    info : PreprocessingInfo
        Información generada durante el preprocesamiento.

    dct_matrix : np.ndarray | None
        Matriz DCT. Si es None, se genera automáticamente.

    Returns
    -------
    reconstructed_image : np.ndarray
        Imagen reconstruida en escala de grises, tipo uint8.
    """
    quantized_blocks = np.asarray(quantized_blocks)

    _validate_blocks_array(quantized_blocks, name="quantized_blocks")
    _validate_preprocessing_info(info)

    _, _, N, _ = quantized_blocks.shape

    if dct_matrix is None:
        dct_matrix = create_dct_matrix(N)

    dequantized_blocks = dequantize_blocks(quantized_blocks, q_matrix)
    reconstructed_shifted_blocks = idct2_blocks(
        dequantized_blocks,
        dct_matrix=dct_matrix
    )

    reconstructed_image = reconstruct_from_blocks(
        reconstructed_shifted_blocks,
        info
    )

    return reconstructed_image


def reconstruct_from_encoded_blocks(
    encoded_blocks: list[EncodedBlock],
    blocks_shape: tuple[int, int, int, int],
    q_matrix: np.ndarray,
    info: PreprocessingInfo,
    dct_matrix: np.ndarray | None = None,
    initial_dc: int = 0
) -> np.ndarray:
    """
    Reconstruye una imagen a partir de los símbolos codificados.

    Esta función aplica el pipeline completo del decodificador:

        símbolos codificados
        -> decodificación RLE + DC diferencial
        -> bloques cuantizados
        -> de-cuantización
        -> IDCT
        -> inverse level shift
        -> reensamblado
        -> crop
        -> uint8

    Parameters
    ----------
    encoded_blocks : list[EncodedBlock]
        Lista de bloques codificados.

    blocks_shape : tuple[int, int, int, int]
        Forma de los bloques cuantizados originales:
        (num_bloques_vertical, num_bloques_horizontal, N, N)

    q_matrix : np.ndarray
        Matriz de cuantización utilizada durante la compresión.

    info : PreprocessingInfo
        Información generada durante el preprocesamiento.

    dct_matrix : np.ndarray | None
        Matriz DCT. Si es None, se genera automáticamente.

    initial_dc : int
        Valor DC inicial usado durante la codificación diferencial.

    Returns
    -------
    reconstructed_image : np.ndarray
        Imagen reconstruida en escala de grises, tipo uint8.
    """
    _validate_preprocessing_info(info)

    quantized_blocks = decode_quantized_blocks(
        encoded_blocks=encoded_blocks,
        blocks_shape=blocks_shape,
        initial_dc=initial_dc
    )

    reconstructed_image = reconstruct_from_quantized_blocks(
        quantized_blocks=quantized_blocks,
        q_matrix=q_matrix,
        info=info,
        dct_matrix=dct_matrix
    )

    return reconstructed_image


def _validate_blocks_array(blocks: np.ndarray, name: str = "blocks") -> None:
    """
    Valida que un arreglo de bloques tenga forma:

        (num_bloques_vertical, num_bloques_horizontal, N, N)

    Parameters
    ----------
    blocks : np.ndarray
        Arreglo a validar.

    name : str
        Nombre descriptivo para mensajes de error.
    """
    if blocks.ndim != 4:
        raise ValueError(
            f"{name} debe tener 4 dimensiones con forma "
            f"(num_bloques_vertical, num_bloques_horizontal, N, N)."
        )

    if blocks.shape[2] != blocks.shape[3]:
        raise ValueError(
            f"Los bloques contenidos en {name} deben ser cuadrados. "
            f"Se recibió forma {blocks.shape}."
        )


def _validate_image_shape_tuple(shape: tuple[int, int], name: str = "shape") -> None:
    """
    Valida una tupla de dimensiones de imagen.

    Parameters
    ----------
    shape : tuple[int, int]
        Tupla esperada: (alto, ancho)

    name : str
        Nombre descriptivo para mensajes de error.
    """
    if not isinstance(shape, tuple):
        raise TypeError(f"{name} debe ser una tupla.")

    if len(shape) != 2:
        raise ValueError(f"{name} debe tener longitud 2: (alto, ancho).")

    height, width = shape

    if height <= 0 or width <= 0:
        raise ValueError(f"Las dimensiones en {name} deben ser positivas.")


def _validate_preprocessing_info(info: PreprocessingInfo) -> None:
    """
    Valida que el objeto PreprocessingInfo tenga dimensiones consistentes.

    Parameters
    ----------
    info : PreprocessingInfo
        Información generada durante el preprocesamiento.
    """
    if not isinstance(info, PreprocessingInfo):
        raise TypeError("info debe ser una instancia de PreprocessingInfo.")

    _validate_image_shape_tuple(info.original_shape, name="info.original_shape")
    _validate_image_shape_tuple(info.padded_shape, name="info.padded_shape")

    original_h, original_w = info.original_shape
    padded_h, padded_w = info.padded_shape

    if padded_h < original_h or padded_w < original_w:
        raise ValueError(
            "Las dimensiones con padding no pueden ser menores que las originales."
        )

    if info.pad_height < 0 or info.pad_width < 0:
        raise ValueError("Los valores de padding no pueden ser negativos.")

    if info.block_size <= 0:
        raise ValueError("info.block_size debe ser positivo.")

    if padded_h % info.block_size != 0 or padded_w % info.block_size != 0:
        raise ValueError(
            "Las dimensiones con padding deben ser múltiplos de info.block_size."
        )


def _validate_padded_shape(actual_shape: tuple[int, int], expected_shape: tuple[int, int]) -> None:
    """
    Verifica que la imagen reensamblada tenga la forma con padding esperada.

    Parameters
    ----------
    actual_shape : tuple[int, int]
        Forma obtenida al reensamblar bloques.

    expected_shape : tuple[int, int]
        Forma esperada según PreprocessingInfo.
    """
    if actual_shape != expected_shape:
        raise ValueError(
            "La forma de la imagen reensamblada no coincide con la forma esperada. "
            f"Se obtuvo {actual_shape}, pero se esperaba {expected_shape}."
        )


if __name__ == "__main__":
    # ============================================================
    # Ejemplo de integración con los módulos anteriores
    # ============================================================

    from preprocessing import preprocess_image
    from dct import dct2_blocks
    from quantization import (
        get_luminance_quantization_matrix,
        scale_quantization_matrix,
        quantize_blocks,
    )
    from coding import encode_quantized_blocks

    image_path = "imagen_prueba.png"
    output_path = "reconstructed_image.png"

    # 1. Preprocesamiento
    blocks, info = preprocess_image(image_path)

    # 2. DCT
    C = create_dct_matrix(info.block_size)
    coeff_blocks = dct2_blocks(blocks, dct_matrix=C)

    # 3. Cuantización
    q_base = get_luminance_quantization_matrix()
    q_matrix = scale_quantization_matrix(q_base, scale=1.0)

    quantized_blocks = quantize_blocks(coeff_blocks, q_matrix)

    # 4. Codificación simbólica
    encoded_blocks = encode_quantized_blocks(quantized_blocks)

    # 5. Reconstrucción desde símbolos codificados
    reconstructed_image = reconstruct_from_encoded_blocks(
        encoded_blocks=encoded_blocks,
        blocks_shape=quantized_blocks.shape,
        q_matrix=q_matrix,
        info=info,
        dct_matrix=C
    )

    # 6. Guardado
    save_grayscale_image(reconstructed_image, output_path)

    print("Reconstrucción completada.")
    print(f"Imagen original: {info.original_shape}")
    print(f"Imagen con padding: {info.padded_shape}")
    print(f"Imagen reconstruida: {reconstructed_image.shape}")
    print(f"Imagen guardada en: {output_path}")