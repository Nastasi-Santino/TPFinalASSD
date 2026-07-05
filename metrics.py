"""
metrics.py

Módulo de métricas para evaluar un compresor de imágenes basado en DCT.

Métricas implementadas:
1. Error cuadrático medio (MSE).
2. Error absoluto medio (MAE).
3. Relación señal a ruido pico (PSNR).
4. Porcentaje de coeficientes nulos.
5. Cantidad total de coeficientes.
6. Cantidad de coeficientes no nulos.
7. Cantidad de símbolos RLE.
8. Cantidad promedio de símbolos RLE por bloque.
9. Estimaciones simples de bits.
10. Relación de compresión estimada.
11. Porcentaje de reducción estimado.

Este módulo no realiza compresión ni reconstrucción. Solo evalúa resultados
ya obtenidos por los módulos anteriores.
"""

from dataclasses import dataclass

import numpy as np

from coding import EncodedBlock, count_rle_symbols, average_rle_symbols_per_block


@dataclass
class QualityMetrics:
    """
    Métricas de calidad de reconstrucción.

    mse:
        Error cuadrático medio entre imagen original y reconstruida.

    mae:
        Error absoluto medio entre imagen original y reconstruida.

    psnr:
        Relación señal a ruido pico, en dB.
    """
    mse: float
    mae: float
    psnr: float


@dataclass
class CompressionMetrics:
    """
    Métricas asociadas a compresión.

    total_coefficients:
        Cantidad total de coeficientes cuantizados.

    zero_coefficients:
        Cantidad de coeficientes iguales a cero.

    nonzero_coefficients:
        Cantidad de coeficientes distintos de cero.

    zero_percentage:
        Porcentaje de coeficientes nulos.

    rle_symbols:
        Cantidad total de símbolos RLE generados para coeficientes AC.

    average_rle_symbols_per_block:
        Cantidad promedio de símbolos RLE por bloque.

    original_bits:
        Estimación de bits de la imagen original.

    compressed_bits:
        Estimación de bits de la representación comprimida.

    compression_ratio:
        Relación de compresión estimada.

    reduction_percentage:
        Porcentaje de reducción estimado.
    """
    total_coefficients: int
    zero_coefficients: int
    nonzero_coefficients: int
    zero_percentage: float
    rle_symbols: int
    average_rle_symbols_per_block: float
    original_bits: int
    compressed_bits: int
    compression_ratio: float
    reduction_percentage: float


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calcula el error cuadrático medio entre dos imágenes.

    Parameters
    ----------
    original : np.ndarray
        Imagen original.

    reconstructed : np.ndarray
        Imagen reconstruida.

    Returns
    -------
    float
        Error cuadrático medio.
    """
    original, reconstructed = _prepare_image_pair(original, reconstructed)

    error = original - reconstructed
    return float(np.mean(error ** 2))


def mae(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calcula el error absoluto medio entre dos imágenes.

    Parameters
    ----------
    original : np.ndarray
        Imagen original.

    reconstructed : np.ndarray
        Imagen reconstruida.

    Returns
    -------
    float
        Error absoluto medio.
    """
    original, reconstructed = _prepare_image_pair(original, reconstructed)

    error = np.abs(original - reconstructed)
    return float(np.mean(error))


def psnr(
    original: np.ndarray,
    reconstructed: np.ndarray,
    max_value: float = 255.0
) -> float:
    """
    Calcula el PSNR entre dos imágenes.

    Para imágenes de 8 bits, max_value = 255.

    Parameters
    ----------
    original : np.ndarray
        Imagen original.

    reconstructed : np.ndarray
        Imagen reconstruida.

    max_value : float
        Valor máximo posible de intensidad.

    Returns
    -------
    float
        PSNR en dB. Si las imágenes son idénticas, devuelve infinito.
    """
    if max_value <= 0:
        raise ValueError("max_value debe ser positivo.")

    mse_value = mse(original, reconstructed)

    if mse_value == 0:
        return float("inf")

    return float(10.0 * np.log10((max_value ** 2) / mse_value))


def compute_quality_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    max_value: float = 255.0
) -> QualityMetrics:
    """
    Calcula un conjunto de métricas de calidad de reconstrucción.

    Parameters
    ----------
    original : np.ndarray
        Imagen original.

    reconstructed : np.ndarray
        Imagen reconstruida.

    max_value : float
        Valor máximo posible de intensidad.

    Returns
    -------
    QualityMetrics
        Dataclass con MSE, MAE y PSNR.
    """
    return QualityMetrics(
        mse=mse(original, reconstructed),
        mae=mae(original, reconstructed),
        psnr=psnr(original, reconstructed, max_value=max_value)
    )


def count_total_coefficients(quantized_blocks: np.ndarray) -> int:
    """
    Cuenta la cantidad total de coeficientes en un arreglo de bloques cuantizados.

    Parameters
    ----------
    quantized_blocks : np.ndarray
        Bloques cuantizados.

    Returns
    -------
    int
        Cantidad total de coeficientes.
    """
    quantized_blocks = np.asarray(quantized_blocks)

    if quantized_blocks.size == 0:
        raise ValueError("quantized_blocks no puede estar vacío.")

    return int(quantized_blocks.size)


def count_zero_coefficients(quantized_blocks: np.ndarray) -> int:
    """
    Cuenta la cantidad de coeficientes nulos.

    Parameters
    ----------
    quantized_blocks : np.ndarray
        Bloques cuantizados.

    Returns
    -------
    int
        Cantidad de coeficientes iguales a cero.
    """
    quantized_blocks = np.asarray(quantized_blocks)

    if quantized_blocks.size == 0:
        raise ValueError("quantized_blocks no puede estar vacío.")

    return int(np.count_nonzero(quantized_blocks == 0))


def count_nonzero_coefficients(quantized_blocks: np.ndarray) -> int:
    """
    Cuenta la cantidad de coeficientes no nulos.

    Parameters
    ----------
    quantized_blocks : np.ndarray
        Bloques cuantizados.

    Returns
    -------
    int
        Cantidad de coeficientes distintos de cero.
    """
    quantized_blocks = np.asarray(quantized_blocks)

    if quantized_blocks.size == 0:
        raise ValueError("quantized_blocks no puede estar vacío.")

    return int(np.count_nonzero(quantized_blocks != 0))


def zero_percentage(quantized_blocks: np.ndarray) -> float:
    """
    Calcula el porcentaje de coeficientes nulos.

    Parameters
    ----------
    quantized_blocks : np.ndarray
        Bloques cuantizados.

    Returns
    -------
    float
        Porcentaje de coeficientes iguales a cero.
    """
    total = count_total_coefficients(quantized_blocks)
    zeros = count_zero_coefficients(quantized_blocks)

    return 100.0 * zeros / total


def estimate_original_bits(
    image_shape: tuple[int, int],
    bits_per_pixel: int = 8
) -> int:
    """
    Estima la cantidad de bits de una imagen original sin comprimir.

    Para una imagen en escala de grises de 8 bits:

        bits = alto * ancho * 8

    Parameters
    ----------
    image_shape : tuple[int, int]
        Forma de la imagen original:
        (alto, ancho)

    bits_per_pixel : int
        Cantidad de bits por píxel.

    Returns
    -------
    int
        Cantidad de bits estimada.
    """
    _validate_image_shape_tuple(image_shape, name="image_shape")

    if bits_per_pixel <= 0:
        raise ValueError("bits_per_pixel debe ser positivo.")

    height, width = image_shape

    return int(height * width * bits_per_pixel)


def estimate_bits_from_nonzero_coefficients(
    quantized_blocks: np.ndarray,
    bits_per_coefficient: int = 16
) -> int:
    """
    Estima bits comprimidos usando solo coeficientes no nulos.

    Esta es una estimación muy simple, útil como referencia inicial.
    Supone que cada coeficiente no nulo se almacena usando un número fijo
    de bits.

    Parameters
    ----------
    quantized_blocks : np.ndarray
        Bloques cuantizados.

    bits_per_coefficient : int
        Bits asignados a cada coeficiente no nulo.

    Returns
    -------
    int
        Estimación de bits comprimidos.
    """
    if bits_per_coefficient <= 0:
        raise ValueError("bits_per_coefficient debe ser positivo.")

    nonzeros = count_nonzero_coefficients(quantized_blocks)

    return int(nonzeros * bits_per_coefficient)


def estimate_bits_from_rle(
    encoded_blocks: list[EncodedBlock],
    bits_per_dc_diff: int = 16,
    bits_per_run_length: int = 6,
    bits_per_amplitude: int = 16,
    bits_per_eob: int = 4
) -> int:
    """
    Estima la cantidad de bits de la representación codificada con RLE.

    Esta función NO implementa un bitstream real. Es una estimación simple
    para comparar configuraciones.

    Modelo utilizado:
    - Cada bloque almacena un DC diferencial usando bits_per_dc_diff bits.
    - Cada par RLE (run_length, amplitude) usa:
        bits_per_run_length + bits_per_amplitude bits.
    - Cada símbolo EOB usa bits_per_eob bits.

    Parameters
    ----------
    encoded_blocks : list[EncodedBlock]
        Lista de bloques codificados.

    bits_per_dc_diff : int
        Bits estimados para cada coeficiente DC diferencial.

    bits_per_run_length : int
        Bits estimados para el run_length de cada símbolo RLE.

    bits_per_amplitude : int
        Bits estimados para la amplitud de cada símbolo RLE.

    bits_per_eob : int
        Bits estimados para el símbolo EOB.

    Returns
    -------
    int
        Cantidad estimada de bits comprimidos.
    """
    _validate_encoded_blocks(encoded_blocks)

    if bits_per_dc_diff <= 0:
        raise ValueError("bits_per_dc_diff debe ser positivo.")

    if bits_per_run_length <= 0:
        raise ValueError("bits_per_run_length debe ser positivo.")

    if bits_per_amplitude <= 0:
        raise ValueError("bits_per_amplitude debe ser positivo.")

    if bits_per_eob <= 0:
        raise ValueError("bits_per_eob debe ser positivo.")

    total_bits = 0

    for block in encoded_blocks:
        total_bits += bits_per_dc_diff

        for symbol in block.ac_symbols:
            if symbol == "EOB":
                total_bits += bits_per_eob
            else:
                total_bits += bits_per_run_length + bits_per_amplitude

    return int(total_bits)


def compression_ratio(original_bits: int, compressed_bits: int) -> float:
    """
    Calcula la relación de compresión.

        CR = original_bits / compressed_bits

    Parameters
    ----------
    original_bits : int
        Cantidad de bits original.

    compressed_bits : int
        Cantidad de bits comprimida.

    Returns
    -------
    float
        Relación de compresión.
    """
    if original_bits <= 0:
        raise ValueError("original_bits debe ser positivo.")

    if compressed_bits <= 0:
        raise ValueError("compressed_bits debe ser positivo.")

    return float(original_bits / compressed_bits)


def reduction_percentage(original_bits: int, compressed_bits: int) -> float:
    """
    Calcula el porcentaje de reducción de tamaño.

        R = (1 - compressed_bits / original_bits) * 100 %

    Parameters
    ----------
    original_bits : int
        Cantidad de bits original.

    compressed_bits : int
        Cantidad de bits comprimida.

    Returns
    -------
    float
        Porcentaje de reducción.
    """
    if original_bits <= 0:
        raise ValueError("original_bits debe ser positivo.")

    if compressed_bits < 0:
        raise ValueError("compressed_bits no puede ser negativo.")

    return float((1.0 - compressed_bits / original_bits) * 100.0)


def compute_compression_metrics(
    quantized_blocks: np.ndarray,
    encoded_blocks: list[EncodedBlock],
    original_image_shape: tuple[int, int],
    bits_per_pixel: int = 8,
    bits_per_dc_diff: int = 16,
    bits_per_run_length: int = 6,
    bits_per_amplitude: int = 16,
    bits_per_eob: int = 4
) -> CompressionMetrics:
    """
    Calcula un conjunto de métricas asociadas a la compresión.

    Parameters
    ----------
    quantized_blocks : np.ndarray
        Bloques cuantizados.

    encoded_blocks : list[EncodedBlock]
        Bloques codificados con DC diferencial y RLE.

    original_image_shape : tuple[int, int]
        Forma original de la imagen:
        (alto, ancho)

    bits_per_pixel : int
        Bits por píxel de la imagen original.

    bits_per_dc_diff : int
        Bits estimados para cada DC diferencial.

    bits_per_run_length : int
        Bits estimados para cada run length.

    bits_per_amplitude : int
        Bits estimados para cada amplitud.

    bits_per_eob : int
        Bits estimados para cada símbolo EOB.

    Returns
    -------
    CompressionMetrics
        Dataclass con métricas de compresión.
    """
    quantized_blocks = np.asarray(quantized_blocks)

    _validate_blocks_array(quantized_blocks, name="quantized_blocks")
    _validate_encoded_blocks(encoded_blocks)

    total = count_total_coefficients(quantized_blocks)
    zeros = count_zero_coefficients(quantized_blocks)
    nonzeros = count_nonzero_coefficients(quantized_blocks)
    zero_pct = zero_percentage(quantized_blocks)

    rle_total = count_rle_symbols(encoded_blocks)
    rle_avg = average_rle_symbols_per_block(encoded_blocks)

    original_bits = estimate_original_bits(
        image_shape=original_image_shape,
        bits_per_pixel=bits_per_pixel
    )

    compressed_bits = estimate_bits_from_rle(
        encoded_blocks=encoded_blocks,
        bits_per_dc_diff=bits_per_dc_diff,
        bits_per_run_length=bits_per_run_length,
        bits_per_amplitude=bits_per_amplitude,
        bits_per_eob=bits_per_eob
    )

    cr = compression_ratio(original_bits, compressed_bits)
    reduction = reduction_percentage(original_bits, compressed_bits)

    return CompressionMetrics(
        total_coefficients=total,
        zero_coefficients=zeros,
        nonzero_coefficients=nonzeros,
        zero_percentage=zero_pct,
        rle_symbols=rle_total,
        average_rle_symbols_per_block=rle_avg,
        original_bits=original_bits,
        compressed_bits=compressed_bits,
        compression_ratio=cr,
        reduction_percentage=reduction
    )


def print_quality_metrics(metrics: QualityMetrics) -> None:
    """
    Imprime métricas de calidad de manera prolija.

    Parameters
    ----------
    metrics : QualityMetrics
        Métricas de calidad.
    """
    print("Métricas de calidad")
    print("-------------------")
    print(f"MSE : {metrics.mse:.4f}")
    print(f"MAE : {metrics.mae:.4f}")

    if np.isinf(metrics.psnr):
        print("PSNR: infinito")
    else:
        print(f"PSNR: {metrics.psnr:.2f} dB")


def print_compression_metrics(metrics: CompressionMetrics) -> None:
    """
    Imprime métricas de compresión de manera prolija.

    Parameters
    ----------
    metrics : CompressionMetrics
        Métricas de compresión.
    """
    print("Métricas de compresión")
    print("----------------------")
    print(f"Coeficientes totales       : {metrics.total_coefficients}")
    print(f"Coeficientes nulos         : {metrics.zero_coefficients}")
    print(f"Coeficientes no nulos      : {metrics.nonzero_coefficients}")
    print(f"Porcentaje de ceros        : {metrics.zero_percentage:.2f}%")
    print(f"Símbolos RLE totales       : {metrics.rle_symbols}")
    print(f"Símbolos RLE promedio/bloque: {metrics.average_rle_symbols_per_block:.2f}")
    print(f"Bits originales estimados  : {metrics.original_bits}")
    print(f"Bits comprimidos estimados : {metrics.compressed_bits}")
    print(f"Relación de compresión     : {metrics.compression_ratio:.2f}:1")
    print(f"Reducción estimada         : {metrics.reduction_percentage:.2f}%")


def _prepare_image_pair(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Valida y convierte dos imágenes para comparar.

    Parameters
    ----------
    original : np.ndarray
        Imagen original.

    reconstructed : np.ndarray
        Imagen reconstruida.

    Returns
    -------
    original_float : np.ndarray
        Imagen original en float64.

    reconstructed_float : np.ndarray
        Imagen reconstruida en float64.
    """
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)

    if original.ndim != 2 or reconstructed.ndim != 2:
        raise ValueError("Las imágenes deben ser 2D en escala de grises.")

    if original.shape != reconstructed.shape:
        raise ValueError(
            "Las imágenes deben tener la misma forma. "
            f"original.shape={original.shape}, reconstructed.shape={reconstructed.shape}"
        )

    return original, reconstructed


def _validate_blocks_array(blocks: np.ndarray, name: str = "blocks") -> None:
    """
    Valida que un arreglo de bloques tenga forma:

        (num_bloques_vertical, num_bloques_horizontal, N, N)

    Parameters
    ----------
    blocks : np.ndarray
        Arreglo de bloques.

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
    Valida una tupla de forma de imagen.

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


def _validate_encoded_blocks(encoded_blocks: list[EncodedBlock]) -> None:
    """
    Valida una lista de bloques codificados.

    Parameters
    ----------
    encoded_blocks : list[EncodedBlock]
        Lista de bloques codificados.
    """
    if not isinstance(encoded_blocks, list):
        raise TypeError("encoded_blocks debe ser una lista.")

    if len(encoded_blocks) == 0:
        raise ValueError("encoded_blocks no puede estar vacío.")

    for block in encoded_blocks:
        if not isinstance(block, EncodedBlock):
            raise TypeError(
                "Todos los elementos de encoded_blocks deben ser instancias de EncodedBlock."
            )


if __name__ == "__main__":
    # ============================================================
    # Ejemplo de integración con el pipeline completo
    # ============================================================

    from preprocessing import preprocess_image, rgb_to_grayscale, load_image
    from dct import create_dct_matrix, dct2_blocks
    from quantization import (
        get_luminance_quantization_matrix,
        scale_quantization_matrix,
        quantize_blocks,
    )
    from coding import encode_quantized_blocks
    from reconstruction import reconstruct_from_encoded_blocks, save_grayscale_image

    image_path = "imagen_prueba.png"
    output_path = "reconstructed_image.png"

    # Imagen original en escala de grises para comparar métricas
    original_rgb = load_image(image_path)
    original_gray = rgb_to_grayscale(original_rgb)

    # 1. Preprocesamiento
    blocks, info = preprocess_image(image_path)

    # 2. DCT
    C = create_dct_matrix(info.block_size)
    coeff_blocks = dct2_blocks(blocks, dct_matrix=C)

    # 3. Cuantización
    q_base = get_luminance_quantization_matrix()
    q_matrix = scale_quantization_matrix(q_base, scale=1.0)
    quantized_blocks = quantize_blocks(coeff_blocks, q_matrix)

    # 4. Codificación
    encoded_blocks = encode_quantized_blocks(quantized_blocks)

    # 5. Reconstrucción
    reconstructed_image = reconstruct_from_encoded_blocks(
        encoded_blocks=encoded_blocks,
        blocks_shape=quantized_blocks.shape,
        q_matrix=q_matrix,
        info=info,
        dct_matrix=C
    )

    save_grayscale_image(reconstructed_image, output_path)

    # 6. Métricas
    quality = compute_quality_metrics(
        original=original_gray,
        reconstructed=reconstructed_image
    )

    compression = compute_compression_metrics(
        quantized_blocks=quantized_blocks,
        encoded_blocks=encoded_blocks,
        original_image_shape=info.original_shape
    )

    print_quality_metrics(quality)
    print()
    print_compression_metrics(compression)