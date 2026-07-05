"""
coding.py

Módulo de reordenamiento y codificación para un compresor de imágenes basado en DCT.

Etapas implementadas:
1. Recorrido zig-zag de bloques 8x8.
2. Recorrido zig-zag inverso.
3. Separación entre coeficiente DC y coeficientes AC.
4. Codificación diferencial de coeficientes DC.
5. Codificación RLE de coeficientes AC.
6. Decodificación RLE.
7. Codificación y decodificación completa de bloques cuantizados.

Este módulo NO implementa codificación Huffman ni escritura de un bitstream real.
La salida es una representación simbólica intermedia, adecuada para analizar
el proceso de compresión y para reconstruir la imagen en etapas posteriores.
"""

from dataclasses import dataclass

import numpy as np


BLOCK_SIZE = 8
EOB_SYMBOL = "EOB"  # End Of Block


@dataclass
class EncodedBlock:
    """
    Representación codificada de un bloque cuantizado.

    dc_diff:
        Diferencia entre el coeficiente DC del bloque actual y el DC del bloque anterior.

    ac_symbols:
        Lista de símbolos RLE para los coeficientes AC.

        Cada símbolo puede ser:
        - una tupla (run_length, amplitude), donde:
            run_length: cantidad de ceros antes del coeficiente no nulo;
            amplitude: valor del coeficiente no nulo.
        - el string "EOB", que indica que el resto del bloque son ceros.
    """
    dc_diff: int
    ac_symbols: list[tuple[int, int] | str]


def create_zigzag_indices(N: int = BLOCK_SIZE) -> list[tuple[int, int]]:
    """
    Genera los índices del recorrido zig-zag para una matriz NxN.

    El recorrido zig-zag ordena los coeficientes aproximadamente desde
    bajas frecuencias hacia altas frecuencias.

    Parameters
    ----------
    N : int
        Tamaño de la matriz cuadrada.

    Returns
    -------
    indices : list[tuple[int, int]]
        Lista de pares (fila, columna) en el orden del recorrido zig-zag.
    """
    if N <= 0:
        raise ValueError("N debe ser positivo.")

    indices = []

    # Cada diagonal tiene suma de índices constante: s = i + j.
    for s in range(2 * N - 1):
        if s % 2 == 0:
            # Diagonales pares: se recorren hacia arriba.
            row_start = min(s, N - 1)
            row_end = max(-1, s - N)

            for i in range(row_start, row_end, -1):
                j = s - i
                indices.append((i, j))
        else:
            # Diagonales impares: se recorren hacia abajo.
            col_start = min(s, N - 1)
            col_end = max(-1, s - N)

            for j in range(col_start, col_end, -1):
                i = s - j
                indices.append((i, j))

    return indices


def zigzag_scan(block: np.ndarray, indices: list[tuple[int, int]] | None = None) -> np.ndarray:
    """
    Convierte un bloque NxN en un vector de longitud N^2 usando recorrido zig-zag.

    Parameters
    ----------
    block : np.ndarray
        Bloque cuadrado de coeficientes cuantizados.

    indices : list[tuple[int, int]] | None
        Índices zig-zag. Si es None, se generan automáticamente.

    Returns
    -------
    vector : np.ndarray
        Vector unidimensional con los coeficientes en orden zig-zag.
    """
    block = np.asarray(block)

    _validate_square_block(block, name="block")

    N = block.shape[0]

    if indices is None:
        indices = create_zigzag_indices(N)
    else:
        _validate_zigzag_indices(indices, N)

    vector = np.array([block[i, j] for i, j in indices], dtype=np.int32)

    return vector


def inverse_zigzag_scan(
    vector: np.ndarray,
    N: int = BLOCK_SIZE,
    indices: list[tuple[int, int]] | None = None
) -> np.ndarray:
    """
    Reconstruye una matriz NxN a partir de un vector en orden zig-zag.

    Parameters
    ----------
    vector : np.ndarray
        Vector de longitud N^2.

    N : int
        Tamaño del bloque cuadrado.

    indices : list[tuple[int, int]] | None
        Índices zig-zag. Si es None, se generan automáticamente.

    Returns
    -------
    block : np.ndarray
        Bloque reconstruido de forma (N, N).
    """
    vector = np.asarray(vector)

    if N <= 0:
        raise ValueError("N debe ser positivo.")

    if vector.ndim != 1:
        raise ValueError("vector debe ser un arreglo unidimensional.")

    if vector.size != N * N:
        raise ValueError(
            f"vector debe tener longitud {N*N}, pero tiene longitud {vector.size}."
        )

    if indices is None:
        indices = create_zigzag_indices(N)
    else:
        _validate_zigzag_indices(indices, N)

    block = np.zeros((N, N), dtype=np.int32)

    for k, (i, j) in enumerate(indices):
        block[i, j] = vector[k]

    return block


def split_dc_ac(zigzag_vector: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Separa el coeficiente DC y los coeficientes AC de un vector zig-zag.

    Parameters
    ----------
    zigzag_vector : np.ndarray
        Vector de coeficientes en orden zig-zag.

    Returns
    -------
    dc : int
        Coeficiente DC, correspondiente al primer elemento del vector.

    ac : np.ndarray
        Coeficientes AC, correspondientes al resto del vector.
    """
    zigzag_vector = np.asarray(zigzag_vector)

    if zigzag_vector.ndim != 1:
        raise ValueError("zigzag_vector debe ser un arreglo unidimensional.")

    if zigzag_vector.size < 2:
        raise ValueError("zigzag_vector debe tener al menos dos elementos.")

    dc = int(zigzag_vector[0])
    ac = zigzag_vector[1:].astype(np.int32)

    return dc, ac


def differential_encode_dc(dc_values: list[int] | np.ndarray, initial_dc: int = 0) -> list[int]:
    """
    Codifica diferencialmente una secuencia de coeficientes DC.

    En lugar de almacenar cada DC directamente, se almacena la diferencia
    respecto del DC anterior:

        dc_diff[n] = dc[n] - dc[n-1]

    Parameters
    ----------
    dc_values : list[int] | np.ndarray
        Secuencia de coeficientes DC.

    initial_dc : int
        Valor de referencia anterior al primer bloque.

    Returns
    -------
    dc_diffs : list[int]
        Secuencia de diferencias DC.
    """
    dc_values = [int(dc) for dc in dc_values]

    dc_diffs = []
    previous_dc = int(initial_dc)

    for dc in dc_values:
        diff = dc - previous_dc
        dc_diffs.append(int(diff))
        previous_dc = dc

    return dc_diffs


def differential_decode_dc(dc_diffs: list[int] | np.ndarray, initial_dc: int = 0) -> list[int]:
    """
    Decodifica una secuencia de diferencias DC.

    Reconstruye los valores DC originales mediante:

        dc[n] = dc_diff[n] + dc[n-1]

    Parameters
    ----------
    dc_diffs : list[int] | np.ndarray
        Secuencia de diferencias DC.

    initial_dc : int
        Valor de referencia anterior al primer bloque.

    Returns
    -------
    dc_values : list[int]
        Secuencia reconstruida de coeficientes DC.
    """
    dc_diffs = [int(diff) for diff in dc_diffs]

    dc_values = []
    previous_dc = int(initial_dc)

    for diff in dc_diffs:
        dc = previous_dc + diff
        dc_values.append(int(dc))
        previous_dc = dc

    return dc_values


def rle_encode_ac(ac_coeffs: np.ndarray) -> list[tuple[int, int] | str]:
    """
    Codifica los coeficientes AC usando Run-Length Encoding.

    El RLE representa secuencias de ceros mediante pares:

        (run_length, amplitude)

    donde:
    - run_length es la cantidad de ceros antes del próximo coeficiente no nulo;
    - amplitude es el valor del coeficiente no nulo.

    Al final se agrega el símbolo EOB si los coeficientes restantes son cero.

    Parameters
    ----------
    ac_coeffs : np.ndarray
        Vector de coeficientes AC.

    Returns
    -------
    symbols : list[tuple[int, int] | str]
        Lista de símbolos RLE.
    """
    ac_coeffs = np.asarray(ac_coeffs, dtype=np.int32)

    if ac_coeffs.ndim != 1:
        raise ValueError("ac_coeffs debe ser un arreglo unidimensional.")

    symbols: list[tuple[int, int] | str] = []
    zero_count = 0

    for coeff in ac_coeffs:
        coeff = int(coeff)

        if coeff == 0:
            zero_count += 1
        else:
            symbols.append((zero_count, coeff))
            zero_count = 0

    # Si quedaron ceros al final, se indica con EOB.
    # En JPEG real, EOB evita almacenar una cola de ceros.
    if zero_count > 0:
        symbols.append(EOB_SYMBOL)

    return symbols


def rle_decode_ac(
    symbols: list[tuple[int, int] | str],
    ac_length: int = BLOCK_SIZE * BLOCK_SIZE - 1
) -> np.ndarray:
    """
    Decodifica una secuencia RLE de coeficientes AC.

    Parameters
    ----------
    symbols : list[tuple[int, int] | str]
        Lista de símbolos RLE.

    ac_length : int
        Longitud esperada del vector AC reconstruido.
        Para bloques 8x8, ac_length = 63.

    Returns
    -------
    ac_coeffs : np.ndarray
        Vector AC reconstruido.
    """
    if ac_length <= 0:
        raise ValueError("ac_length debe ser positivo.")

    ac_coeffs = []

    for symbol in symbols:
        if symbol == EOB_SYMBOL:
            break

        if not _is_valid_rle_pair(symbol):
            raise ValueError(f"Símbolo RLE inválido: {symbol}")

        run_length, amplitude = symbol

        ac_coeffs.extend([0] * int(run_length))
        ac_coeffs.append(int(amplitude))

        if len(ac_coeffs) > ac_length:
            raise ValueError(
                "La secuencia RLE decodificada supera la longitud AC esperada."
            )

    # Se completa con ceros hasta llegar a la longitud esperada.
    while len(ac_coeffs) < ac_length:
        ac_coeffs.append(0)

    return np.array(ac_coeffs, dtype=np.int32)


def encode_quantized_blocks(
    quantized_blocks: np.ndarray,
    initial_dc: int = 0
) -> list[EncodedBlock]:
    """
    Codifica un arreglo de bloques cuantizados.

    Para cada bloque:
    1. Se aplica recorrido zig-zag.
    2. Se separa DC y AC.
    3. El DC se codifica diferencialmente respecto del bloque anterior.
    4. Los AC se codifican con RLE.

    El orden de procesamiento de bloques es raster scan:
    primero de izquierda a derecha y luego de arriba hacia abajo.

    Parameters
    ----------
    quantized_blocks : np.ndarray
        Arreglo de bloques cuantizados con forma:
        (num_bloques_vertical, num_bloques_horizontal, N, N)

    initial_dc : int
        Valor DC de referencia para el primer bloque.

    Returns
    -------
    encoded_blocks : list[EncodedBlock]
        Lista de bloques codificados.
    """
    quantized_blocks = np.asarray(quantized_blocks, dtype=np.int32)

    _validate_blocks_array(quantized_blocks, name="quantized_blocks")

    num_blocks_h, num_blocks_w, N, _ = quantized_blocks.shape
    indices = create_zigzag_indices(N)

    encoded_blocks: list[EncodedBlock] = []
    previous_dc = int(initial_dc)

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            vector = zigzag_scan(quantized_blocks[i, j], indices=indices)
            dc, ac = split_dc_ac(vector)

            dc_diff = dc - previous_dc
            previous_dc = dc

            ac_symbols = rle_encode_ac(ac)

            encoded_blocks.append(
                EncodedBlock(
                    dc_diff=int(dc_diff),
                    ac_symbols=ac_symbols
                )
            )

    return encoded_blocks


def decode_quantized_blocks(
    encoded_blocks: list[EncodedBlock],
    blocks_shape: tuple[int, int, int, int],
    initial_dc: int = 0
) -> np.ndarray:
    """
    Decodifica una lista de bloques codificados y reconstruye los bloques cuantizados.

    Este proceso invierte la función encode_quantized_blocks.

    Parameters
    ----------
    encoded_blocks : list[EncodedBlock]
        Lista de bloques codificados.

    blocks_shape : tuple[int, int, int, int]
        Forma esperada del arreglo de bloques:
        (num_bloques_vertical, num_bloques_horizontal, N, N)

    initial_dc : int
        Valor DC de referencia para el primer bloque.

    Returns
    -------
    quantized_blocks : np.ndarray
        Arreglo reconstruido de bloques cuantizados.
    """
    _validate_blocks_shape_tuple(blocks_shape)

    num_blocks_h, num_blocks_w, N, _ = blocks_shape
    expected_num_blocks = num_blocks_h * num_blocks_w

    if len(encoded_blocks) != expected_num_blocks:
        raise ValueError(
            f"Se esperaban {expected_num_blocks} bloques codificados, "
            f"pero se recibieron {len(encoded_blocks)}."
        )

    indices = create_zigzag_indices(N)
    quantized_blocks = np.zeros(blocks_shape, dtype=np.int32)

    previous_dc = int(initial_dc)
    block_index = 0

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            encoded_block = encoded_blocks[block_index]

            if not isinstance(encoded_block, EncodedBlock):
                raise TypeError(
                    "Todos los elementos de encoded_blocks deben ser EncodedBlock."
                )

            dc = previous_dc + int(encoded_block.dc_diff)
            previous_dc = dc

            ac = rle_decode_ac(
                encoded_block.ac_symbols,
                ac_length=N * N - 1
            )

            vector = np.concatenate(([dc], ac)).astype(np.int32)
            block = inverse_zigzag_scan(vector, N=N, indices=indices)

            quantized_blocks[i, j] = block
            block_index += 1

    return quantized_blocks


def count_rle_symbols(encoded_blocks: list[EncodedBlock]) -> int:
    """
    Cuenta la cantidad total de símbolos RLE utilizados para codificar los AC.

    Parameters
    ----------
    encoded_blocks : list[EncodedBlock]
        Lista de bloques codificados.

    Returns
    -------
    int
        Cantidad total de símbolos AC codificados.
    """
    return sum(len(block.ac_symbols) for block in encoded_blocks)


def average_rle_symbols_per_block(encoded_blocks: list[EncodedBlock]) -> float:
    """
    Calcula la cantidad promedio de símbolos RLE por bloque.

    Parameters
    ----------
    encoded_blocks : list[EncodedBlock]
        Lista de bloques codificados.

    Returns
    -------
    float
        Promedio de símbolos RLE por bloque.
    """
    if len(encoded_blocks) == 0:
        raise ValueError("encoded_blocks no puede estar vacío.")

    return count_rle_symbols(encoded_blocks) / len(encoded_blocks)


def count_eob_symbols(encoded_blocks: list[EncodedBlock]) -> int:
    """
    Cuenta cuántos bloques terminaron con símbolo EOB.

    Parameters
    ----------
    encoded_blocks : list[EncodedBlock]
        Lista de bloques codificados.

    Returns
    -------
    int
        Cantidad de bloques cuya codificación AC termina en EOB.
    """
    count = 0

    for block in encoded_blocks:
        if len(block.ac_symbols) > 0 and block.ac_symbols[-1] == EOB_SYMBOL:
            count += 1

    return count


def _validate_square_block(block: np.ndarray, name: str = "block") -> None:
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


def _validate_blocks_shape_tuple(blocks_shape: tuple[int, int, int, int]) -> None:
    """
    Valida la tupla de forma esperada para un arreglo de bloques.

    Parameters
    ----------
    blocks_shape : tuple[int, int, int, int]
        Forma esperada:
        (num_bloques_vertical, num_bloques_horizontal, N, N)
    """
    if not isinstance(blocks_shape, tuple):
        raise TypeError("blocks_shape debe ser una tupla.")

    if len(blocks_shape) != 4:
        raise ValueError(
            "blocks_shape debe tener longitud 4: "
            "(num_bloques_vertical, num_bloques_horizontal, N, N)."
        )

    num_blocks_h, num_blocks_w, rows, cols = blocks_shape

    if num_blocks_h <= 0 or num_blocks_w <= 0:
        raise ValueError("La cantidad de bloques debe ser positiva.")

    if rows <= 0 or cols <= 0:
        raise ValueError("El tamaño de bloque debe ser positivo.")

    if rows != cols:
        raise ValueError("Los bloques deben ser cuadrados.")


def _validate_zigzag_indices(indices: list[tuple[int, int]], N: int) -> None:
    """
    Valida que una lista de índices zig-zag sea compatible con una matriz NxN.

    Parameters
    ----------
    indices : list[tuple[int, int]]
        Lista de índices a validar.

    N : int
        Tamaño del bloque cuadrado.
    """
    if len(indices) != N * N:
        raise ValueError(
            f"La lista de índices debe tener longitud {N*N}, "
            f"pero tiene longitud {len(indices)}."
        )

    seen = set()

    for idx in indices:
        if not isinstance(idx, tuple) or len(idx) != 2:
            raise ValueError(f"Índice inválido: {idx}")

        i, j = idx

        if not (0 <= i < N and 0 <= j < N):
            raise ValueError(f"Índice fuera de rango para N={N}: {idx}")

        if idx in seen:
            raise ValueError(f"Índice repetido en el recorrido zig-zag: {idx}")

        seen.add(idx)


def _is_valid_rle_pair(symbol: object) -> bool:
    """
    Verifica si un símbolo tiene la forma válida de un par RLE.

    Parameters
    ----------
    symbol : object
        Símbolo a verificar.

    Returns
    -------
    bool
        True si el símbolo es una tupla (run_length, amplitude) válida.
    """
    if not isinstance(symbol, tuple):
        return False

    if len(symbol) != 2:
        return False

    run_length, amplitude = symbol

    if not isinstance(run_length, int):
        return False

    if not isinstance(amplitude, int):
        return False

    if run_length < 0:
        return False

    return True


if __name__ == "__main__":
    # ============================================================
    # Ejemplo mínimo de uso y verificación del módulo
    # ============================================================

    rng = np.random.default_rng(seed=0)

    # Simulamos bloques cuantizados.
    # Usamos muchos ceros para parecerse a la salida real de la cuantización.
    quantized_blocks = rng.integers(
        low=-5,
        high=6,
        size=(3, 4, BLOCK_SIZE, BLOCK_SIZE)
    ).astype(np.int32)

    mask = rng.random(size=quantized_blocks.shape) < 0.85
    quantized_blocks[mask] = 0

    encoded_blocks = encode_quantized_blocks(quantized_blocks)

    decoded_blocks = decode_quantized_blocks(
        encoded_blocks,
        blocks_shape=quantized_blocks.shape
    )

    error = np.max(np.abs(quantized_blocks - decoded_blocks))

    print("Módulo de reordenamiento y codificación")
    print("---------------------------------------")
    print(f"Forma de bloques cuantizados: {quantized_blocks.shape}")
    print(f"Cantidad de bloques codificados: {len(encoded_blocks)}")
    print(f"Cantidad total de símbolos RLE AC: {count_rle_symbols(encoded_blocks)}")
    print(f"Símbolos RLE promedio por bloque: {average_rle_symbols_per_block(encoded_blocks):.2f}")
    print(f"Bloques terminados con EOB: {count_eob_symbols(encoded_blocks)}")
    print(f"Error máximo luego de codificar y decodificar: {error}")

    print("\nEjemplo de primer bloque codificado:")
    print(f"DC diferencial: {encoded_blocks[0].dc_diff}")
    print(f"Símbolos AC: {encoded_blocks[0].ac_symbols}")