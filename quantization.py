"""
quantization.py

Módulo de cuantización para un compresor de imágenes basado en DCT.

Etapas implementadas:
1. Definición de la matriz de cuantización de luminancia tipo JPEG.
2. Escalado de la matriz de cuantización mediante un factor configurable.
3. Generación de matriz de cuantización a partir de un factor de calidad tipo JPEG.
4. Cuantización de un bloque o de un conjunto de bloques.
5. De-cuantización de un bloque o de un conjunto de bloques.

Este módulo trabaja sobre coeficientes DCT. Por lo tanto, se espera que la entrada
a la cuantización sea la salida del módulo dct.py.

No se utilizan librerías de compresión de imágenes. Solo se usa NumPy para
operaciones numéricas.
"""

import numpy as np


BLOCK_SIZE = 8


JPEG_LUMINANCE_QUANTIZATION_MATRIX = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64
)


def get_luminance_quantization_matrix(dtype=np.float64) -> np.ndarray:
    """
    Devuelve una copia de la matriz de cuantización de luminancia tipo JPEG.

    Esta matriz asigna pasos de cuantización menores a las bajas frecuencias
    y pasos mayores a las altas frecuencias.

    Parameters
    ----------
    dtype : data-type
        Tipo de dato deseado para la matriz devuelta.

    Returns
    -------
    q_matrix : np.ndarray
        Matriz de cuantización de tamaño 8x8.
    """
    return JPEG_LUMINANCE_QUANTIZATION_MATRIX.astype(dtype).copy()


def scale_quantization_matrix(
    q_matrix: np.ndarray,
    scale: float = 1.0,
    min_value: int = 1,
    max_value: int = 255,
    dtype=np.float64
) -> np.ndarray:
    """
    Escala una matriz de cuantización mediante un factor multiplicativo.

    Se implementa el modelo simple:

        Q_s = scale * Q

    donde:
    - scale < 1 produce cuantización más suave;
    - scale = 1 conserva la matriz original;
    - scale > 1 produce cuantización más agresiva.

    Luego del escalado, los valores se redondean y se saturan al intervalo
    [min_value, max_value]. Esto evita pasos de cuantización nulos.

    Parameters
    ----------
    q_matrix : np.ndarray
        Matriz de cuantización base.

    scale : float
        Factor de escala positivo.

    min_value : int
        Valor mínimo permitido para un paso de cuantización.

    max_value : int
        Valor máximo permitido para un paso de cuantización.

    dtype : data-type
        Tipo de dato de la matriz resultante.

    Returns
    -------
    scaled_matrix : np.ndarray
        Matriz de cuantización escalada.
    """
    q_matrix = np.asarray(q_matrix, dtype=np.float64)
    _validate_quantization_matrix(q_matrix)

    if scale <= 0:
        raise ValueError("El factor de escala debe ser positivo.")

    if min_value <= 0:
        raise ValueError("min_value debe ser positivo para evitar divisiones por cero.")

    if max_value < min_value:
        raise ValueError("max_value debe ser mayor o igual que min_value.")

    scaled_matrix = np.rint(scale * q_matrix)
    scaled_matrix = np.clip(scaled_matrix, min_value, max_value)

    return scaled_matrix.astype(dtype)


def quality_to_scale(quality: int) -> float:
    """
    Convierte un factor de calidad tipo JPEG en un factor de escala.

    El parámetro quality toma valores entre 1 y 100:
    - quality = 100 implica cuantización muy suave;
    - quality = 50 reproduce aproximadamente la matriz base;
    - quality bajo implica cuantización más agresiva.

    La fórmula utilizada es la habitual en implementaciones JPEG:

        si quality < 50:
            scale = 5000 / quality
        si quality >= 50:
            scale = 200 - 2 * quality

    Luego, este valor se interpreta como porcentaje y se divide por 100.

    Parameters
    ----------
    quality : int
        Factor de calidad entre 1 y 100.

    Returns
    -------
    scale : float
        Factor de escala equivalente.
    """
    if not isinstance(quality, int):
        raise TypeError("quality debe ser un entero.")

    if not (1 <= quality <= 100):
        raise ValueError("quality debe estar en el intervalo [1, 100].")

    if quality < 50:
        scale_percent = 5000 / quality
    else:
        scale_percent = 200 - 2 * quality

    scale = scale_percent / 100.0

    # Para quality=100, la fórmula da scale=0.
    # Se usa un valor pequeño positivo para evitar una matriz nula.
    return max(scale, 0.01)


def create_quality_quantization_matrix(
    quality: int,
    base_matrix: np.ndarray | None = None,
    min_value: int = 1,
    max_value: int = 255,
    dtype=np.float64
) -> np.ndarray:
    """
    Genera una matriz de cuantización a partir de un factor de calidad tipo JPEG.

    Parameters
    ----------
    quality : int
        Factor de calidad entre 1 y 100.

    base_matrix : np.ndarray | None
        Matriz de cuantización base. Si es None, se usa la matriz de luminancia JPEG.

    min_value : int
        Valor mínimo permitido para un paso de cuantización.

    max_value : int
        Valor máximo permitido para un paso de cuantización.

    dtype : data-type
        Tipo de dato de la matriz resultante.

    Returns
    -------
    q_matrix : np.ndarray
        Matriz de cuantización escalada según el factor de calidad.
    """
    if base_matrix is None:
        base_matrix = get_luminance_quantization_matrix(dtype=np.float64)

    scale = quality_to_scale(quality)

    return scale_quantization_matrix(
        q_matrix=base_matrix,
        scale=scale,
        min_value=min_value,
        max_value=max_value,
        dtype=dtype
    )


def quantize_block(coeffs: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    """
    Cuantiza un bloque de coeficientes DCT.

    Para cada coeficiente se aplica:

        F_q[u, v] = round( F[u, v] / Q[u, v] )

    Parameters
    ----------
    coeffs : np.ndarray
        Bloque de coeficientes DCT de forma (N, N).

    q_matrix : np.ndarray
        Matriz de cuantización de forma (N, N).

    Returns
    -------
    quantized : np.ndarray
        Bloque cuantizado con valores enteros.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    q_matrix = np.asarray(q_matrix, dtype=np.float64)

    _validate_block(coeffs, name="coeffs")
    _validate_quantization_matrix(q_matrix, expected_size=coeffs.shape[0])

    quantized = np.rint(coeffs / q_matrix)

    return quantized.astype(np.int32)


def dequantize_block(quantized_coeffs: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    """
    De-cuantiza un bloque de coeficientes.

    Para cada coeficiente se aplica:

        F_tilde[u, v] = F_q[u, v] * Q[u, v]

    Esta operación no recupera exactamente los coeficientes originales,
    ya que la cuantización previa introdujo redondeo.

    Parameters
    ----------
    quantized_coeffs : np.ndarray
        Bloque de coeficientes cuantizados de forma (N, N).

    q_matrix : np.ndarray
        Matriz de cuantización de forma (N, N).

    Returns
    -------
    dequantized : np.ndarray
        Bloque de-cuantizado en formato float64.
    """
    quantized_coeffs = np.asarray(quantized_coeffs)
    q_matrix = np.asarray(q_matrix, dtype=np.float64)

    _validate_block(quantized_coeffs, name="quantized_coeffs")
    _validate_quantization_matrix(q_matrix, expected_size=quantized_coeffs.shape[0])

    dequantized = quantized_coeffs.astype(np.float64) * q_matrix

    return dequantized


def quantize_blocks(coeff_blocks: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    """
    Cuantiza un conjunto de bloques de coeficientes DCT.

    Se espera un arreglo con forma:

        (num_bloques_vertical, num_bloques_horizontal, N, N)

    Parameters
    ----------
    coeff_blocks : np.ndarray
        Bloques de coeficientes DCT.

    q_matrix : np.ndarray
        Matriz de cuantización de forma (N, N).

    Returns
    -------
    quantized_blocks : np.ndarray
        Bloques cuantizados con valores enteros.
    """
    coeff_blocks = np.asarray(coeff_blocks, dtype=np.float64)
    q_matrix = np.asarray(q_matrix, dtype=np.float64)

    _validate_blocks_array(coeff_blocks, name="coeff_blocks")

    _, _, rows, cols = coeff_blocks.shape
    if rows != cols:
        raise ValueError("Cada bloque debe ser cuadrado.")

    _validate_quantization_matrix(q_matrix, expected_size=rows)

    quantized_blocks = np.rint(coeff_blocks / q_matrix)

    return quantized_blocks.astype(np.int32)


def dequantize_blocks(quantized_blocks: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    """
    De-cuantiza un conjunto de bloques cuantizados.

    Se espera un arreglo con forma:

        (num_bloques_vertical, num_bloques_horizontal, N, N)

    Parameters
    ----------
    quantized_blocks : np.ndarray
        Bloques de coeficientes cuantizados.

    q_matrix : np.ndarray
        Matriz de cuantización de forma (N, N).

    Returns
    -------
    dequantized_blocks : np.ndarray
        Bloques de-cuantizados en formato float64.
    """
    quantized_blocks = np.asarray(quantized_blocks)
    q_matrix = np.asarray(q_matrix, dtype=np.float64)

    _validate_blocks_array(quantized_blocks, name="quantized_blocks")

    _, _, rows, cols = quantized_blocks.shape
    if rows != cols:
        raise ValueError("Cada bloque debe ser cuadrado.")

    _validate_quantization_matrix(q_matrix, expected_size=rows)

    dequantized_blocks = quantized_blocks.astype(np.float64) * q_matrix

    return dequantized_blocks


def count_zero_coefficients(quantized_blocks: np.ndarray) -> int:
    """
    Cuenta la cantidad total de coeficientes nulos en un arreglo de bloques cuantizados.

    Parameters
    ----------
    quantized_blocks : np.ndarray
        Bloques cuantizados.

    Returns
    -------
    int
        Cantidad total de coeficientes iguales a cero.
    """
    quantized_blocks = np.asarray(quantized_blocks)

    return int(np.count_nonzero(quantized_blocks == 0))


def zero_percentage(quantized_blocks: np.ndarray) -> float:
    """
    Calcula el porcentaje de coeficientes nulos en un arreglo de bloques cuantizados.

    Esta métrica es útil para estimar el potencial de compresión antes de aplicar
    RLE o codificación entrópica.

    Parameters
    ----------
    quantized_blocks : np.ndarray
        Bloques cuantizados.

    Returns
    -------
    float
        Porcentaje de coeficientes iguales a cero.
    """
    quantized_blocks = np.asarray(quantized_blocks)

    total_coefficients = quantized_blocks.size
    if total_coefficients == 0:
        raise ValueError("El arreglo de bloques no puede estar vacío.")

    zeros = count_zero_coefficients(quantized_blocks)

    return 100.0 * zeros / total_coefficients


def _validate_block(block: np.ndarray, name: str = "block") -> None:
    """
    Valida que un bloque sea una matriz cuadrada 2D.

    Parameters
    ----------
    block : np.ndarray
        Bloque a validar.

    name : str
        Nombre descriptivo para mensajes de error.
    """
    if block.ndim != 2:
        raise ValueError(f"{name} debe ser una matriz 2D.")

    rows, cols = block.shape
    if rows != cols:
        raise ValueError(f"{name} debe ser cuadrado. Se recibió forma {block.shape}.")


def _validate_blocks_array(blocks: np.ndarray, name: str = "blocks") -> None:
    """
    Valida que un arreglo de bloques tenga forma:

        (num_bloques_vertical, num_bloques_horizontal, N, N)

    Parameters
    ----------
    blocks : np.ndarray
        Arreglo de bloques a validar.

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


def _validate_quantization_matrix(
    q_matrix: np.ndarray,
    expected_size: int | None = None
) -> None:
    """
    Valida que una matriz de cuantización sea cuadrada y no tenga valores nulos.

    Parameters
    ----------
    q_matrix : np.ndarray
        Matriz de cuantización.

    expected_size : int | None
        Si no es None, exige que la matriz tenga forma
        (expected_size, expected_size).
    """
    if q_matrix.ndim != 2:
        raise ValueError("La matriz de cuantización debe ser 2D.")

    rows, cols = q_matrix.shape

    if rows != cols:
        raise ValueError("La matriz de cuantización debe ser cuadrada.")

    if expected_size is not None and rows != expected_size:
        raise ValueError(
            f"La matriz de cuantización debe tener forma "
            f"({expected_size}, {expected_size}), pero tiene forma {q_matrix.shape}."
        )

    if np.any(q_matrix <= 0):
        raise ValueError(
            "Todos los elementos de la matriz de cuantización deben ser positivos."
        )


if __name__ == "__main__":
    # ============================================================
    # Ejemplo mínimo de uso y verificación del módulo
    # ============================================================

    rng = np.random.default_rng(seed=0)

    # Simulamos coeficientes DCT de varios bloques
    coeff_blocks = rng.normal(
        loc=0.0,
        scale=80.0,
        size=(4, 5, BLOCK_SIZE, BLOCK_SIZE)
    )

    # Matriz base
    q_base = get_luminance_quantization_matrix()

    # Matriz escalada simple
    q_scaled = scale_quantization_matrix(q_base, scale=1.0)

    # Matriz a partir de factor de calidad
    q_quality_80 = create_quality_quantization_matrix(quality=80)

    quantized_blocks = quantize_blocks(coeff_blocks, q_scaled)
    dequantized_blocks = dequantize_blocks(quantized_blocks, q_scaled)

    print("Módulo de cuantización")
    print("----------------------")
    print(f"Forma de bloques DCT: {coeff_blocks.shape}")
    print(f"Forma de bloques cuantizados: {quantized_blocks.shape}")
    print(f"Forma de bloques de-cuantizados: {dequantized_blocks.shape}")

    print("\nMatriz de cuantización base:")
    print(q_base.astype(int))

    print("\nMatriz de cuantización con quality=80:")
    print(q_quality_80.astype(int))

    print("\nEstadísticas de cuantización:")
    print(f"Coeficientes nulos: {count_zero_coefficients(quantized_blocks)}")
    print(f"Porcentaje de ceros: {zero_percentage(quantized_blocks):.2f}%")

    quantization_error = coeff_blocks - dequantized_blocks
    print(f"Error medio absoluto de cuantización: {np.mean(np.abs(quantization_error)):.4f}")
    print(f"Error máximo absoluto de cuantización: {np.max(np.abs(quantization_error)):.4f}")