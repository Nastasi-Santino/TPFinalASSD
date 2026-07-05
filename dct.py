"""
dct.py

Módulo para calcular la Transformada Discreta del Coseno (DCT) y su inversa
sobre bloques de imagen.

Se implementa la DCT ortonormal de tipo II, que es la utilizada en compresión
de imágenes tipo JPEG. Gracias a la ortonormalidad, la transformada inversa
puede calcularse reutilizando la misma matriz de transformación y tomando su
transpuesta.

Este módulo incluye:
1. Generación de la matriz DCT ortonormal.
2. DCT 1D e IDCT 1D.
3. DCT 2D e IDCT 2D para un bloque.
4. Aplicación de DCT e IDCT sobre un conjunto de bloques.

No se utilizan funciones de alto nivel de bibliotecas externas como scipy.fftpack.dct.
Solo se utiliza NumPy para operaciones matriciales.
"""

import numpy as np


BLOCK_SIZE = 8


def alpha(k: int, N: int) -> float:
    """
    Factor de normalización de la DCT ortonormal.

    Parameters
    ----------
    k : int
        Índice de frecuencia.

    N : int
        Longitud de la señal o tamaño del bloque.

    Returns
    -------
    float
        Factor de normalización correspondiente.
    """
    if N <= 0:
        raise ValueError("N debe ser positivo.")

    if not (0 <= k < N):
        raise ValueError(f"El índice k debe cumplir 0 <= k < {N}.")

    if k == 0:
        return np.sqrt(1.0 / N)
    return np.sqrt(2.0 / N)


def create_dct_matrix(N: int = BLOCK_SIZE, dtype=np.float64) -> np.ndarray:
    """
    Genera la matriz ortonormal de la DCT tipo II de tamaño N x N.

    La matriz C se define como:

        C[k, n] = alpha(k) * cos( pi/N * (n + 1/2) * k )

    donde alpha(k) es el factor de normalización ortonormal.

    Parameters
    ----------
    N : int
        Tamaño de la matriz DCT.

    dtype : data-type
        Tipo de dato del arreglo resultante.

    Returns
    -------
    C : np.ndarray
        Matriz DCT ortonormal de forma (N, N).
    """
    if N <= 0:
        raise ValueError("El tamaño N debe ser positivo.")

    C = np.zeros((N, N), dtype=np.float64)

    for k in range(N):
        for n in range(N):
            C[k, n] = alpha(k, N) * np.cos((np.pi / N) * (n + 0.5) * k)

    return C.astype(dtype)


def is_orthonormal(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    """
    Verifica si una matriz es ortonormal dentro de una tolerancia dada.

    Una matriz ortonormal cumple:

        M @ M.T = I

    Parameters
    ----------
    matrix : np.ndarray
        Matriz a verificar.

    atol : float
        Tolerancia absoluta para la comparación numérica.

    Returns
    -------
    bool
        True si la matriz es ortonormal dentro de la tolerancia indicada.
    """
    if matrix.ndim != 2:
        raise ValueError("La matriz debe ser bidimensional.")

    rows, cols = matrix.shape
    if rows != cols:
        return False

    identity = np.eye(rows, dtype=matrix.dtype)
    product = matrix @ matrix.T

    return np.allclose(product, identity, atol=atol)


def dct_1d(x: np.ndarray, dct_matrix: np.ndarray | None = None) -> np.ndarray:
    """
    Aplica la DCT 1D ortonormal a un vector.

    Si x es un vector columna o fila de longitud N:

        X = C @ x

    donde C es la matriz DCT ortonormal.

    Parameters
    ----------
    x : np.ndarray
        Vector de entrada de longitud N.

    dct_matrix : np.ndarray | None
        Matriz DCT de tamaño (N, N). Si es None, se genera automáticamente.

    Returns
    -------
    X : np.ndarray
        Vector transformado.
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("La entrada x debe ser un vector unidimensional.")

    N = x.shape[0]

    if dct_matrix is None:
        dct_matrix = create_dct_matrix(N)
    else:
        _validate_square_matrix(dct_matrix, expected_size=N, name="dct_matrix")

    return dct_matrix @ x


def idct_1d(X: np.ndarray, dct_matrix: np.ndarray | None = None) -> np.ndarray:
    """
    Aplica la transformada inversa de la DCT 1D ortonormal.

    Como la matriz DCT es ortonormal:

        x = C.T @ X

    Parameters
    ----------
    X : np.ndarray
        Vector transformado de longitud N.

    dct_matrix : np.ndarray | None
        Matriz DCT de tamaño (N, N). Si es None, se genera automáticamente.

    Returns
    -------
    x : np.ndarray
        Vector reconstruido.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("La entrada X debe ser un vector unidimensional.")

    N = X.shape[0]

    if dct_matrix is None:
        dct_matrix = create_dct_matrix(N)
    else:
        _validate_square_matrix(dct_matrix, expected_size=N, name="dct_matrix")

    return dct_matrix.T @ X


def dct2_block(block: np.ndarray, dct_matrix: np.ndarray | None = None) -> np.ndarray:
    """
    Aplica la DCT 2D ortonormal a un bloque 2D.

    Para un bloque B de tamaño N x N:

        F = C @ B @ C.T

    donde C es la matriz DCT ortonormal.

    Parameters
    ----------
    block : np.ndarray
        Bloque de entrada de forma (N, N).

    dct_matrix : np.ndarray | None
        Matriz DCT de tamaño (N, N). Si es None, se genera automáticamente.

    Returns
    -------
    coeffs : np.ndarray
        Coeficientes DCT del bloque.
    """
    block = np.asarray(block, dtype=np.float64)

    if block.ndim != 2:
        raise ValueError("El bloque debe ser una matriz 2D.")

    rows, cols = block.shape
    if rows != cols:
        raise ValueError("El bloque debe ser cuadrado.")

    N = rows

    if dct_matrix is None:
        dct_matrix = create_dct_matrix(N)
    else:
        _validate_square_matrix(dct_matrix, expected_size=N, name="dct_matrix")

    return dct_matrix @ block @ dct_matrix.T


def idct2_block(coeffs: np.ndarray, dct_matrix: np.ndarray | None = None) -> np.ndarray:
    """
    Aplica la transformada inversa de la DCT 2D ortonormal a un bloque.

    Para una matriz de coeficientes F de tamaño N x N:

        B = C.T @ F @ C

    donde C es la matriz DCT ortonormal.

    Parameters
    ----------
    coeffs : np.ndarray
        Matriz de coeficientes DCT de forma (N, N).

    dct_matrix : np.ndarray | None
        Matriz DCT de tamaño (N, N). Si es None, se genera automáticamente.

    Returns
    -------
    block : np.ndarray
        Bloque reconstruido en el dominio espacial.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)

    if coeffs.ndim != 2:
        raise ValueError("Los coeficientes deben ser una matriz 2D.")

    rows, cols = coeffs.shape
    if rows != cols:
        raise ValueError("La matriz de coeficientes debe ser cuadrada.")

    N = rows

    if dct_matrix is None:
        dct_matrix = create_dct_matrix(N)
    else:
        _validate_square_matrix(dct_matrix, expected_size=N, name="dct_matrix")

    return dct_matrix.T @ coeffs @ dct_matrix


def dct2_blocks(blocks: np.ndarray, dct_matrix: np.ndarray | None = None) -> np.ndarray:
    """
    Aplica la DCT 2D a un conjunto de bloques.

    Se espera un arreglo con forma:

        (num_bloques_vertical, num_bloques_horizontal, N, N)

    Parameters
    ----------
    blocks : np.ndarray
        Arreglo de bloques espaciales.

    dct_matrix : np.ndarray | None
        Matriz DCT de tamaño (N, N). Si es None, se genera automáticamente.

    Returns
    -------
    coeff_blocks : np.ndarray
        Arreglo de bloques transformados, con la misma forma que blocks.
    """
    blocks = np.asarray(blocks, dtype=np.float64)

    _validate_blocks_array(blocks, name="blocks")

    _, _, rows, cols = blocks.shape
    if rows != cols:
        raise ValueError("Cada bloque debe ser cuadrado.")

    N = rows

    if dct_matrix is None:
        dct_matrix = create_dct_matrix(N)
    else:
        _validate_square_matrix(dct_matrix, expected_size=N, name="dct_matrix")

    num_blocks_h, num_blocks_w = blocks.shape[0], blocks.shape[1]
    coeff_blocks = np.zeros_like(blocks, dtype=np.float64)

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            coeff_blocks[i, j] = dct2_block(blocks[i, j], dct_matrix=dct_matrix)

    return coeff_blocks


def idct2_blocks(coeff_blocks: np.ndarray, dct_matrix: np.ndarray | None = None) -> np.ndarray:
    """
    Aplica la transformada inversa de la DCT 2D a un conjunto de bloques.

    Se espera un arreglo con forma:

        (num_bloques_vertical, num_bloques_horizontal, N, N)

    Parameters
    ----------
    coeff_blocks : np.ndarray
        Arreglo de bloques en el dominio DCT.

    dct_matrix : np.ndarray | None
        Matriz DCT de tamaño (N, N). Si es None, se genera automáticamente.

    Returns
    -------
    blocks : np.ndarray
        Arreglo de bloques reconstruidos en el dominio espacial.
    """
    coeff_blocks = np.asarray(coeff_blocks, dtype=np.float64)

    _validate_blocks_array(coeff_blocks, name="coeff_blocks")

    _, _, rows, cols = coeff_blocks.shape
    if rows != cols:
        raise ValueError("Cada bloque debe ser cuadrado.")

    N = rows

    if dct_matrix is None:
        dct_matrix = create_dct_matrix(N)
    else:
        _validate_square_matrix(dct_matrix, expected_size=N, name="dct_matrix")

    num_blocks_h, num_blocks_w = coeff_blocks.shape[0], coeff_blocks.shape[1]
    blocks = np.zeros_like(coeff_blocks, dtype=np.float64)

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            blocks[i, j] = idct2_block(coeff_blocks[i, j], dct_matrix=dct_matrix)

    return blocks


def _validate_square_matrix(matrix: np.ndarray, expected_size: int | None = None, name: str = "matrix") -> None:
    """
    Valida que una matriz sea cuadrada y, opcionalmente, de un tamaño esperado.

    Parameters
    ----------
    matrix : np.ndarray
        Matriz a validar.

    expected_size : int | None
        Si no es None, exige que la matriz tenga forma (expected_size, expected_size).

    name : str
        Nombre descriptivo para mensajes de error.
    """
    if matrix.ndim != 2:
        raise ValueError(f"{name} debe ser una matriz 2D.")

    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"{name} debe ser cuadrada.")

    if expected_size is not None and rows != expected_size:
        raise ValueError(
            f"{name} debe tener forma ({expected_size}, {expected_size}) "
            f"pero tiene forma {matrix.shape}."
        )


def _validate_blocks_array(blocks: np.ndarray, name: str = "blocks") -> None:
    """
    Valida que el arreglo de bloques tenga forma:

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


if __name__ == "__main__":
    # ============================================================
    # Ejemplo mínimo de uso y verificación de reconstrucción
    # ============================================================

    N = 8
    C = create_dct_matrix(N)

    print("Matriz DCT generada.")
    print(f"Forma: {C.shape}")
    print(f"¿Es ortonormal?: {is_orthonormal(C)}")

    # Bloque de prueba
    rng = np.random.default_rng(seed=0)
    test_block = rng.integers(low=-128, high=128, size=(N, N)).astype(np.float64)

    coeffs = dct2_block(test_block, dct_matrix=C)
    reconstructed_block = idct2_block(coeffs, dct_matrix=C)

    reconstruction_error = np.max(np.abs(test_block - reconstructed_block))

    print("\nPrueba de DCT 2D + IDCT 2D")
    print(f"Error máximo absoluto de reconstrucción: {reconstruction_error:.12e}")

    # Prueba con un arreglo de bloques
    test_blocks = rng.integers(low=-128, high=128, size=(4, 5, N, N)).astype(np.float64)
    coeff_blocks = dct2_blocks(test_blocks, dct_matrix=C)
    reconstructed_blocks = idct2_blocks(coeff_blocks, dct_matrix=C)

    blocks_error = np.max(np.abs(test_blocks - reconstructed_blocks))

    print("\nPrueba sobre múltiples bloques")
    print(f"Forma de los bloques: {test_blocks.shape}")
    print(f"Error máximo absoluto de reconstrucción: {blocks_error:.12e}")