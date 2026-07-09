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

JPEG_CHROMINANCE_QUANTIZATION_MATRIX = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
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

def get_chrominance_quantization_matrix(dtype=np.float64) -> np.ndarray:
    """
    Devuelve una copia de la matriz de cuantización de crominancia tipo JPEG.

    Esta matriz se utiliza para las componentes Cb y Cr. En general es más
    agresiva que la matriz de luminancia, ya que el sistema visual humano es
    menos sensible a errores de alta frecuencia en crominancia.

    Parameters
    ----------
    dtype : data-type
        Tipo de dato deseado para la matriz devuelta.

    Returns
    -------
    q_matrix : np.ndarray
        Matriz de cuantización de crominancia de tamaño 8x8.
    """
    return JPEG_CHROMINANCE_QUANTIZATION_MATRIX.astype(dtype).copy()

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

def get_color_quantization_matrices(
    scale: float = 1.0,
    chroma_scale: float | None = None,
    min_value: int = 1,
    max_value: int = 255,
    dtype=np.float64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve las matrices de cuantización para una imagen color en YCbCr.

    La componente Y usa la matriz de luminancia, mientras que Cb y Cr usan
    la matriz de crominancia.

    Parameters
    ----------
    scale : float
        Factor de escala aplicado a la matriz de luminancia.

    chroma_scale : float | None
        Factor de escala aplicado a las matrices de crominancia.
        Si es None, se usa el mismo valor que scale.

    min_value : int
        Valor mínimo permitido para los pasos de cuantización.

    max_value : int
        Valor máximo permitido para los pasos de cuantización.

    dtype : data-type
        Tipo de dato de las matrices resultantes.

    Returns
    -------
    q_y : np.ndarray
        Matriz de cuantización para luminancia Y.

    q_cb : np.ndarray
        Matriz de cuantización para crominancia Cb.

    q_cr : np.ndarray
        Matriz de cuantización para crominancia Cr.
    """
    if chroma_scale is None:
        chroma_scale = scale

    q_y_base = get_luminance_quantization_matrix(dtype=np.float64)
    q_c_base = get_chrominance_quantization_matrix(dtype=np.float64)

    q_y = scale_quantization_matrix(
        q_matrix=q_y_base,
        scale=scale,
        min_value=min_value,
        max_value=max_value,
        dtype=dtype
    )

    q_cb = scale_quantization_matrix(
        q_matrix=q_c_base,
        scale=chroma_scale,
        min_value=min_value,
        max_value=max_value,
        dtype=dtype
    )

    q_cr = scale_quantization_matrix(
        q_matrix=q_c_base,
        scale=chroma_scale,
        min_value=min_value,
        max_value=max_value,
        dtype=dtype
    )

    return q_y, q_cb, q_cr

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



# ============================================================
# Cuantización adaptativa por bloque
# ============================================================

SMOOTH_BLOCK = 0
EDGE_BLOCK = 1
TEXTURE_BLOCK = 2

BLOCK_CLASS_NAMES = {
    SMOOTH_BLOCK: "smooth",
    EDGE_BLOCK: "edge",
    TEXTURE_BLOCK: "texture",
}

DEFAULT_SMOOTH_SCALE = 1.5
DEFAULT_EDGE_SCALE = 0.75
DEFAULT_TEXTURE_SCALE = 1.0

SOBEL_X_KERNEL = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    dtype=np.float64
)

SOBEL_Y_KERNEL = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ],
    dtype=np.float64
)


def convolve2d_same_edge(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Aplica una convolución 2D con padding por replicación de borde.

    Esta función se usa para implementar filtros FIR 2D simples, como Sobel,
    sin depender de librerías externas.

    Parameters
    ----------
    image : np.ndarray
        Imagen 2D de entrada.

    kernel : np.ndarray
        Kernel FIR 2D.

    Returns
    -------
    output : np.ndarray
        Imagen filtrada con la misma forma que la entrada.
    """
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    _validate_block(image, name="image")

    if kernel.ndim != 2:
        raise ValueError("kernel debe ser una matriz 2D.")

    kernel_height, kernel_width = kernel.shape

    if kernel_height <= 0 or kernel_width <= 0:
        raise ValueError("kernel debe tener dimensiones positivas.")

    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    padded = np.pad(
        image,
        pad_width=((pad_h, pad_h), (pad_w, pad_w)),
        mode="edge"
    )

    output = np.zeros_like(image, dtype=np.float64)

    # Convolución: se invierte el kernel en ambas dimensiones.
    flipped_kernel = np.flipud(np.fliplr(kernel))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * flipped_kernel)

    return output


def compute_block_features(block: np.ndarray, epsilon: float = 1e-12) -> dict[str, float]:
    """
    Calcula características espaciales de un bloque.

    Las características calculadas son:
    - media;
    - varianza;
    - energía de gradiente horizontal;
    - energía de gradiente vertical;
    - energía total de gradiente;
    - direccionalidad.

    Parameters
    ----------
    block : np.ndarray
        Bloque espacial 2D, típicamente de tamaño 8x8.

    epsilon : float
        Constante pequeña para evitar divisiones por cero.

    Returns
    -------
    features : dict[str, float]
        Diccionario con las características del bloque.
    """
    block = np.asarray(block, dtype=np.float64)

    _validate_block(block, name="block")

    if epsilon <= 0:
        raise ValueError("epsilon debe ser positivo.")

    mean_value = float(np.mean(block))
    variance = float(np.var(block))

    gx = convolve2d_same_edge(block, SOBEL_X_KERNEL)
    gy = convolve2d_same_edge(block, SOBEL_Y_KERNEL)

    gradient_energy_x = float(np.mean(gx ** 2))
    gradient_energy_y = float(np.mean(gy ** 2))
    gradient_energy = gradient_energy_x + gradient_energy_y

    directionality = abs(gradient_energy_x - gradient_energy_y) / (
        gradient_energy + epsilon
    )

    return {
        "mean": mean_value,
        "variance": variance,
        "gradient_energy_x": gradient_energy_x,
        "gradient_energy_y": gradient_energy_y,
        "gradient_energy": gradient_energy,
        "directionality": float(directionality),
    }


def compute_blocks_features(blocks: np.ndarray) -> dict[str, np.ndarray]:
    """
    Calcula características espaciales para un arreglo de bloques.

    Parameters
    ----------
    blocks : np.ndarray
        Arreglo de bloques con forma:
        (num_bloques_vertical, num_bloques_horizontal, N, N)

    Returns
    -------
    features : dict[str, np.ndarray]
        Diccionario con mapas 2D de características.
    """
    blocks = np.asarray(blocks, dtype=np.float64)

    _validate_blocks_array(blocks, name="blocks")

    num_blocks_h, num_blocks_w, _, _ = blocks.shape

    means = np.zeros((num_blocks_h, num_blocks_w), dtype=np.float64)
    variances = np.zeros((num_blocks_h, num_blocks_w), dtype=np.float64)
    gradient_energy_x = np.zeros((num_blocks_h, num_blocks_w), dtype=np.float64)
    gradient_energy_y = np.zeros((num_blocks_h, num_blocks_w), dtype=np.float64)
    gradient_energy = np.zeros((num_blocks_h, num_blocks_w), dtype=np.float64)
    directionality = np.zeros((num_blocks_h, num_blocks_w), dtype=np.float64)

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block_features = compute_block_features(blocks[i, j])

            means[i, j] = block_features["mean"]
            variances[i, j] = block_features["variance"]
            gradient_energy_x[i, j] = block_features["gradient_energy_x"]
            gradient_energy_y[i, j] = block_features["gradient_energy_y"]
            gradient_energy[i, j] = block_features["gradient_energy"]
            directionality[i, j] = block_features["directionality"]

    return {
        "mean": means,
        "variance": variances,
        "gradient_energy_x": gradient_energy_x,
        "gradient_energy_y": gradient_energy_y,
        "gradient_energy": gradient_energy,
        "directionality": directionality,
    }


def classify_blocks_by_content(
    spatial_blocks: np.ndarray,
    variance_threshold: float | None = None,
    gradient_threshold: float | None = None,
    directionality_threshold: float = 0.35,
    smooth_variance_percentile: float = 35.0,
    gradient_percentile: float = 55.0,
    return_features: bool = False
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Clasifica bloques espaciales como suaves, con bordes o con textura.

    La clasificación se basa en:
    - varianza local;
    - energía de gradiente;
    - direccionalidad del gradiente.

    Parameters
    ----------
    spatial_blocks : np.ndarray
        Bloques espaciales antes de aplicar level shift y DCT.
        Forma:
        (num_bloques_vertical, num_bloques_horizontal, N, N)

    variance_threshold : float | None
        Umbral de varianza para detectar bloques suaves. Si es None,
        se estima automáticamente con un percentil.

    gradient_threshold : float | None
        Umbral de energía de gradiente. Si es None, se estima automáticamente
        con un percentil.

    directionality_threshold : float
        Umbral de direccionalidad para distinguir bordes de texturas.

    smooth_variance_percentile : float
        Percentil usado para estimar variance_threshold si no se provee.

    gradient_percentile : float
        Percentil usado para estimar gradient_threshold si no se provee.

    return_features : bool
        Si es True, devuelve también los mapas de características.

    Returns
    -------
    class_map : np.ndarray
        Mapa 2D de clases:
        0 -> bloque suave,
        1 -> bloque con borde,
        2 -> bloque con textura.

    features : dict[str, np.ndarray]
        Solo si return_features=True. Mapas de características por bloque.
    """
    spatial_blocks = np.asarray(spatial_blocks, dtype=np.float64)

    _validate_blocks_array(spatial_blocks, name="spatial_blocks")

    if not (0.0 <= directionality_threshold <= 1.0):
        raise ValueError("directionality_threshold debe estar entre 0 y 1.")

    _validate_percentile(smooth_variance_percentile, "smooth_variance_percentile")
    _validate_percentile(gradient_percentile, "gradient_percentile")

    features = compute_blocks_features(spatial_blocks)

    variance_map = features["variance"]
    gradient_map = features["gradient_energy"]
    directionality_map = features["directionality"]

    if variance_threshold is None:
        variance_threshold = float(np.percentile(variance_map, smooth_variance_percentile))

    if gradient_threshold is None:
        gradient_threshold = float(np.percentile(gradient_map, gradient_percentile))

    if variance_threshold < 0:
        raise ValueError("variance_threshold no puede ser negativo.")

    if gradient_threshold < 0:
        raise ValueError("gradient_threshold no puede ser negativo.")

    class_map = np.full(variance_map.shape, TEXTURE_BLOCK, dtype=np.int32)

    smooth_mask = (
        (variance_map <= variance_threshold)
        & (gradient_map <= gradient_threshold)
    )

    edge_mask = (
        (gradient_map > gradient_threshold)
        & (directionality_map >= directionality_threshold)
    )

    class_map[smooth_mask] = SMOOTH_BLOCK
    class_map[edge_mask] = EDGE_BLOCK

    # Todo lo que no sea suave ni borde queda como textura.
    class_map[(~smooth_mask) & (~edge_mask)] = TEXTURE_BLOCK

    if return_features:
        return class_map, features

    return class_map


def create_scale_map_from_class_map(
    class_map: np.ndarray,
    smooth_scale: float = DEFAULT_SMOOTH_SCALE,
    edge_scale: float = DEFAULT_EDGE_SCALE,
    texture_scale: float = DEFAULT_TEXTURE_SCALE
) -> np.ndarray:
    """
    Convierte un mapa de clases en un mapa de factores de cuantización.

    Parameters
    ----------
    class_map : np.ndarray
        Mapa 2D de clases de bloque.

    smooth_scale : float
        Factor para bloques suaves. Debe ser mayor que 1 para cuantizar
        más agresivamente.

    edge_scale : float
        Factor para bloques con bordes. Debe ser menor que 1 para conservar
        más coeficientes.

    texture_scale : float
        Factor para bloques con textura.

    Returns
    -------
    scale_map : np.ndarray
        Mapa 2D de factores de escala.
    """
    class_map = np.asarray(class_map, dtype=np.int32)

    _validate_class_map(class_map)

    _validate_positive_scale(smooth_scale, "smooth_scale")
    _validate_positive_scale(edge_scale, "edge_scale")
    _validate_positive_scale(texture_scale, "texture_scale")

    scale_map = np.zeros(class_map.shape, dtype=np.float64)

    scale_map[class_map == SMOOTH_BLOCK] = smooth_scale
    scale_map[class_map == EDGE_BLOCK] = edge_scale
    scale_map[class_map == TEXTURE_BLOCK] = texture_scale

    return scale_map


def create_adaptive_quantization_matrices(
    q_matrix: np.ndarray,
    scale_map: np.ndarray,
    min_value: int = 1,
    max_value: int = 255
) -> np.ndarray:
    """
    Crea una matriz de cuantización efectiva por bloque.

    Parameters
    ----------
    q_matrix : np.ndarray
        Matriz de cuantización base de forma (N, N).

    scale_map : np.ndarray
        Mapa 2D de factores con forma
        (num_bloques_vertical, num_bloques_horizontal).

    min_value : int
        Valor mínimo permitido.

    max_value : int
        Valor máximo permitido.

    Returns
    -------
    adaptive_q_matrices : np.ndarray
        Matrices de cuantización por bloque con forma:
        (num_bloques_vertical, num_bloques_horizontal, N, N)
    """
    q_matrix = np.asarray(q_matrix, dtype=np.float64)
    scale_map = np.asarray(scale_map, dtype=np.float64)

    _validate_quantization_matrix(q_matrix)
    _validate_scale_map(scale_map)

    if min_value <= 0:
        raise ValueError("min_value debe ser positivo.")

    if max_value < min_value:
        raise ValueError("max_value debe ser mayor o igual que min_value.")

    adaptive_q_matrices = scale_map[:, :, None, None] * q_matrix[None, None, :, :]
    adaptive_q_matrices = np.rint(adaptive_q_matrices)
    adaptive_q_matrices = np.clip(adaptive_q_matrices, min_value, max_value)

    return adaptive_q_matrices.astype(np.float64)


def quantize_blocks_adaptive(
    coeff_blocks: np.ndarray,
    q_matrix: np.ndarray,
    scale_map: np.ndarray,
    min_value: int = 1,
    max_value: int = 255
) -> np.ndarray:
    """
    Cuantiza bloques DCT usando un factor de escala distinto por bloque.

    Parameters
    ----------
    coeff_blocks : np.ndarray
        Bloques de coeficientes DCT con forma:
        (num_bloques_vertical, num_bloques_horizontal, N, N)

    q_matrix : np.ndarray
        Matriz de cuantización base.

    scale_map : np.ndarray
        Mapa 2D de factores de escala.

    min_value : int
        Valor mínimo permitido para la matriz efectiva.

    max_value : int
        Valor máximo permitido para la matriz efectiva.

    Returns
    -------
    quantized_blocks : np.ndarray
        Bloques cuantizados.
    """
    coeff_blocks = np.asarray(coeff_blocks, dtype=np.float64)
    q_matrix = np.asarray(q_matrix, dtype=np.float64)
    scale_map = np.asarray(scale_map, dtype=np.float64)

    _validate_blocks_array(coeff_blocks, name="coeff_blocks")
    _validate_quantization_matrix(q_matrix, expected_size=coeff_blocks.shape[2])
    _validate_scale_map_for_blocks(scale_map, coeff_blocks)

    adaptive_q = create_adaptive_quantization_matrices(
        q_matrix=q_matrix,
        scale_map=scale_map,
        min_value=min_value,
        max_value=max_value
    )

    quantized_blocks = np.rint(coeff_blocks / adaptive_q)

    return quantized_blocks.astype(np.int32)


def dequantize_blocks_adaptive(
    quantized_blocks: np.ndarray,
    q_matrix: np.ndarray,
    scale_map: np.ndarray,
    min_value: int = 1,
    max_value: int = 255
) -> np.ndarray:
    """
    De-cuantiza bloques usando el mismo mapa de escalas adaptativo.

    Parameters
    ----------
    quantized_blocks : np.ndarray
        Bloques cuantizados.

    q_matrix : np.ndarray
        Matriz de cuantización base.

    scale_map : np.ndarray
        Mapa 2D de factores de escala usado en la cuantización.

    min_value : int
        Valor mínimo permitido para la matriz efectiva.

    max_value : int
        Valor máximo permitido para la matriz efectiva.

    Returns
    -------
    dequantized_blocks : np.ndarray
        Bloques de-cuantizados.
    """
    quantized_blocks = np.asarray(quantized_blocks)
    q_matrix = np.asarray(q_matrix, dtype=np.float64)
    scale_map = np.asarray(scale_map, dtype=np.float64)

    _validate_blocks_array(quantized_blocks, name="quantized_blocks")
    _validate_quantization_matrix(q_matrix, expected_size=quantized_blocks.shape[2])
    _validate_scale_map_for_blocks(scale_map, quantized_blocks)

    adaptive_q = create_adaptive_quantization_matrices(
        q_matrix=q_matrix,
        scale_map=scale_map,
        min_value=min_value,
        max_value=max_value
    )

    return quantized_blocks.astype(np.float64) * adaptive_q


def count_block_classes(class_map: np.ndarray) -> dict[str, int]:
    """
    Cuenta cuántos bloques hay de cada clase.

    Parameters
    ----------
    class_map : np.ndarray
        Mapa de clases.

    Returns
    -------
    counts : dict[str, int]
        Cantidad de bloques suaves, con bordes y con textura.
    """
    class_map = np.asarray(class_map, dtype=np.int32)

    _validate_class_map(class_map)

    return {
        "smooth": int(np.count_nonzero(class_map == SMOOTH_BLOCK)),
        "edge": int(np.count_nonzero(class_map == EDGE_BLOCK)),
        "texture": int(np.count_nonzero(class_map == TEXTURE_BLOCK)),
    }


def print_block_class_summary(class_map: np.ndarray) -> None:
    """
    Imprime un resumen del mapa de clases de bloques.
    """
    counts = count_block_classes(class_map)
    total = sum(counts.values())

    print("Clasificación adaptativa de bloques")
    print("-----------------------------------")

    for class_name in ("smooth", "edge", "texture"):
        count = counts[class_name]
        percentage = 100.0 * count / total if total > 0 else 0.0
        print(f"{class_name:>8}: {count:5d} bloques ({percentage:6.2f}%)")


def _validate_percentile(value: float, name: str) -> None:
    """
    Valida un percentil.
    """
    if not (0.0 <= value <= 100.0):
        raise ValueError(f"{name} debe estar entre 0 y 100.")


def _validate_positive_scale(scale: float, name: str) -> None:
    """
    Valida un factor de escala positivo.
    """
    if scale <= 0:
        raise ValueError(f"{name} debe ser positivo.")


def _validate_class_map(class_map: np.ndarray) -> None:
    """
    Valida un mapa de clases de bloque.
    """
    if class_map.ndim != 2:
        raise ValueError("class_map debe ser una matriz 2D.")

    valid_classes = {SMOOTH_BLOCK, EDGE_BLOCK, TEXTURE_BLOCK}
    observed_classes = set(np.unique(class_map).tolist())

    if not observed_classes.issubset(valid_classes):
        raise ValueError(
            f"class_map contiene clases inválidas: {observed_classes}. "
            f"Clases válidas: {valid_classes}."
        )


def _validate_scale_map(scale_map: np.ndarray) -> None:
    """
    Valida un mapa de factores de escala.
    """
    if scale_map.ndim != 2:
        raise ValueError("scale_map debe ser una matriz 2D.")

    if np.any(scale_map <= 0):
        raise ValueError("Todos los factores de scale_map deben ser positivos.")


def _validate_scale_map_for_blocks(scale_map: np.ndarray, blocks: np.ndarray) -> None:
    """
    Valida que un scale_map sea compatible con un arreglo de bloques.
    """
    _validate_scale_map(scale_map)

    expected_shape = blocks.shape[:2]

    if scale_map.shape != expected_shape:
        raise ValueError(
            "scale_map debe tener la misma cantidad de bloques que blocks. "
            f"Se esperaba {expected_shape}, pero se recibió {scale_map.shape}."
        )

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