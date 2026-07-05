"""
main.py

Programa principal para ejecutar un compresor de imágenes basado en DCT.

Pipeline implementado:

1. Carga y preprocesamiento de imagen.
2. Conversión a escala de grises.
3. Padding por replicación de borde.
4. División en bloques de 8x8.
5. Corrimiento de nivel.
6. DCT 2D por bloques.
7. Cuantización.
8. Zig-zag + DC diferencial + RLE.
9. Decodificación simbólica.
10. De-cuantización.
11. IDCT.
12. Reconstrucción de imagen.
13. Cálculo de métricas.
14. Barrido opcional de factores de cuantización.

Este archivo no implementa las etapas internas del compresor, sino que usa
los módulos desarrollados:
- preprocessing.py
- dct.py
- quantization.py
- coding.py
- reconstruction.py
- metrics.py
"""

from dataclasses import asdict
from pathlib import Path
import argparse
import csv

import numpy as np

from preprocessing import (
    preprocess_image,
    load_image,
    rgb_to_grayscale,
)

from dct import (
    create_dct_matrix,
    dct2_blocks,
)

from quantization import (
    get_luminance_quantization_matrix,
    scale_quantization_matrix,
    create_quality_quantization_matrix,
    quantize_blocks,
)

from coding import (
    encode_quantized_blocks,
)

from reconstruction import (
    reconstruct_from_encoded_blocks,
    save_grayscale_image,
)

from metrics import (
    compute_quality_metrics,
    compute_compression_metrics,
    print_quality_metrics,
    print_compression_metrics,
)


def run_compression(
    image_path: str | Path,
    output_dir: str | Path,
    scale: float = 1.0,
    quality: int | None = None,
    save_image: bool = True,
) -> dict:
    """
    Ejecuta el pipeline completo de compresión y reconstrucción para una imagen.

    Se puede controlar la cuantización de dos maneras:
    - usando scale, donde Q_s = scale * Q;
    - usando quality, con un factor tipo JPEG entre 1 y 100.

    Si quality no es None, se ignora scale.

    Parameters
    ----------
    image_path : str | Path
        Ruta de la imagen de entrada.

    output_dir : str | Path
        Carpeta donde se guardarán los resultados.

    scale : float
        Factor de escala simple para la matriz de cuantización.

    quality : int | None
        Factor de calidad tipo JPEG. Si es None, se usa scale.

    save_image : bool
        Si True, guarda la imagen reconstruida.

    Returns
    -------
    results : dict
        Diccionario con métricas y datos principales de la ejecución.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Imagen original para métricas
    # ============================================================
    original_rgb = load_image(image_path)
    original_gray = rgb_to_grayscale(original_rgb)

    # ============================================================
    # 1. Preprocesamiento
    # ============================================================
    blocks, info = preprocess_image(image_path)

    # ============================================================
    # 2. DCT
    # ============================================================
    C = create_dct_matrix(info.block_size)
    coeff_blocks = dct2_blocks(blocks, dct_matrix=C)

    # ============================================================
    # 3. Cuantización
    # ============================================================
    q_base = get_luminance_quantization_matrix()

    if quality is not None:
        q_matrix = create_quality_quantization_matrix(quality=quality)
        quantization_label = f"quality_{quality}"
    else:
        q_matrix = scale_quantization_matrix(q_base, scale=scale)
        quantization_label = f"scale_{scale:g}"

    quantized_blocks = quantize_blocks(coeff_blocks, q_matrix)

    # ============================================================
    # 4. Codificación simbólica
    # ============================================================
    encoded_blocks = encode_quantized_blocks(quantized_blocks)

    # ============================================================
    # 5. Reconstrucción
    # ============================================================
    reconstructed_image = reconstruct_from_encoded_blocks(
        encoded_blocks=encoded_blocks,
        blocks_shape=quantized_blocks.shape,
        q_matrix=q_matrix,
        info=info,
        dct_matrix=C,
    )

    # ============================================================
    # 6. Métricas
    # ============================================================
    quality_metrics = compute_quality_metrics(
        original=original_gray,
        reconstructed=reconstructed_image,
    )

    compression_metrics = compute_compression_metrics(
        quantized_blocks=quantized_blocks,
        encoded_blocks=encoded_blocks,
        original_image_shape=info.original_shape,
    )

    # ============================================================
    # 7. Guardado de imagen reconstruida
    # ============================================================
    output_image_path = None

    if save_image:
        output_image_path = output_dir / f"reconstructed_{quantization_label}.png"
        save_grayscale_image(reconstructed_image, output_image_path)

    # ============================================================
    # 8. Empaquetado de resultados
    # ============================================================
    results = {
        "image_path": str(image_path),
        "output_image_path": str(output_image_path) if output_image_path else "",
        "scale": scale,
        "quality": quality if quality is not None else "",
        "original_height": info.original_shape[0],
        "original_width": info.original_shape[1],
        "padded_height": info.padded_shape[0],
        "padded_width": info.padded_shape[1],
        "pad_height": info.pad_height,
        "pad_width": info.pad_width,
        "num_blocks_vertical": quantized_blocks.shape[0],
        "num_blocks_horizontal": quantized_blocks.shape[1],
    }

    results.update(asdict(quality_metrics))
    results.update(asdict(compression_metrics))

    return results


def run_scale_sweep(
    image_path: str | Path,
    output_dir: str | Path,
    scales: list[float],
) -> list[dict]:
    """
    Ejecuta el compresor para varios factores de cuantización.

    Parameters
    ----------
    image_path : str | Path
        Ruta de la imagen de entrada.

    output_dir : str | Path
        Carpeta donde se guardarán resultados.

    scales : list[float]
        Lista de factores de cuantización.

    Returns
    -------
    all_results : list[dict]
        Lista de resultados, uno por cada factor de escala.
    """
    all_results = []

    for scale in scales:
        print("=" * 60)
        print(f"Ejecutando compresión con scale = {scale}")
        print("=" * 60)

        results = run_compression(
            image_path=image_path,
            output_dir=output_dir,
            scale=scale,
            quality=None,
            save_image=True,
        )

        all_results.append(results)

        print_summary_from_results(results)
        print()

    return all_results


def run_quality_sweep(
    image_path: str | Path,
    output_dir: str | Path,
    qualities: list[int],
) -> list[dict]:
    """
    Ejecuta el compresor para varios factores de calidad tipo JPEG.

    Parameters
    ----------
    image_path : str | Path
        Ruta de la imagen de entrada.

    output_dir : str | Path
        Carpeta donde se guardarán resultados.

    qualities : list[int]
        Lista de factores de calidad entre 1 y 100.

    Returns
    -------
    all_results : list[dict]
        Lista de resultados, uno por cada calidad.
    """
    all_results = []

    for quality in qualities:
        print("=" * 60)
        print(f"Ejecutando compresión con quality = {quality}")
        print("=" * 60)

        results = run_compression(
            image_path=image_path,
            output_dir=output_dir,
            quality=quality,
            save_image=True,
        )

        all_results.append(results)

        print_summary_from_results(results)
        print()

    return all_results


def save_results_csv(results: list[dict], output_path: str | Path) -> None:
    """
    Guarda una lista de resultados en formato CSV.

    Parameters
    ----------
    results : list[dict]
        Lista de diccionarios de métricas.

    output_path : str | Path
        Ruta del archivo CSV de salida.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(results) == 0:
        raise ValueError("No hay resultados para guardar.")

    fieldnames = list(results[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_plots(results: list[dict], output_dir: str | Path) -> None:
    """
    Guarda gráficos simples de desempeño.

    Genera:
    - PSNR vs scale o quality.
    - Compression ratio vs scale o quality.
    - Zero percentage vs scale o quality.

    Parameters
    ----------
    results : list[dict]
        Lista de resultados.

    output_dir : str | Path
        Carpeta donde se guardarán los gráficos.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no está instalado. No se generarán gráficos.")
        return

    if len(results) == 0:
        return

    using_quality = results[0]["quality"] != ""

    if using_quality:
        x = [int(r["quality"]) for r in results]
        x_label = "Factor de calidad"
        suffix = "quality"
    else:
        x = [float(r["scale"]) for r in results]
        x_label = "Factor de escala"
        suffix = "scale"

    psnr_values = [float(r["psnr"]) for r in results]
    cr_values = [float(r["compression_ratio"]) for r in results]
    zero_values = [float(r["zero_percentage"]) for r in results]

    # PSNR
    plt.figure(figsize=(7, 5))
    plt.plot(x, psnr_values, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("PSNR [dB]")
    plt.title("Calidad de reconstrucción")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"psnr_vs_{suffix}.pdf", bbox_inches="tight")
    plt.savefig(output_dir / f"psnr_vs_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Relación de compresión
    plt.figure(figsize=(7, 5))
    plt.plot(x, cr_values, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("Relación de compresión")
    plt.title("Relación de compresión estimada")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"compression_ratio_vs_{suffix}.pdf", bbox_inches="tight")
    plt.savefig(output_dir / f"compression_ratio_vs_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Porcentaje de ceros
    plt.figure(figsize=(7, 5))
    plt.plot(x, zero_values, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("Coeficientes nulos [%]")
    plt.title("Porcentaje de coeficientes nulos")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"zeros_vs_{suffix}.pdf", bbox_inches="tight")
    plt.savefig(output_dir / f"zeros_vs_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()


def print_summary_from_results(results: dict) -> None:
    """
    Imprime un resumen compacto de resultados.

    Parameters
    ----------
    results : dict
        Diccionario de resultados.
    """
    print("Resumen de ejecución")
    print("--------------------")
    print(f"Imagen original        : {results['original_height']} x {results['original_width']}")
    print(f"Imagen con padding     : {results['padded_height']} x {results['padded_width']}")
    print(f"Padding agregado       : filas={results['pad_height']}, columnas={results['pad_width']}")
    print(f"Bloques                : {results['num_blocks_vertical']} x {results['num_blocks_horizontal']}")

    if results["quality"] != "":
        print(f"Factor de calidad      : {results['quality']}")
    else:
        print(f"Factor de escala       : {results['scale']}")

    print()
    print(f"MSE                    : {results['mse']:.4f}")
    print(f"MAE                    : {results['mae']:.4f}")

    if np.isinf(results["psnr"]):
        print("PSNR                   : infinito")
    else:
        print(f"PSNR                   : {results['psnr']:.2f} dB")

    print(f"Coeficientes nulos     : {results['zero_percentage']:.2f}%")
    print(f"Símbolos RLE totales   : {results['rle_symbols']}")
    print(f"Símbolos RLE/bloque    : {results['average_rle_symbols_per_block']:.2f}")
    print(f"Bits originales        : {results['original_bits']}")
    print(f"Bits comprimidos       : {results['compressed_bits']}")
    print(f"Relación de compresión : {results['compression_ratio']:.2f}:1")
    print(f"Reducción estimada     : {results['reduction_percentage']:.2f}%")

    if results["output_image_path"]:
        print(f"Imagen reconstruida    : {results['output_image_path']}")


def parse_float_list(text: str) -> list[float]:
    """
    Convierte un string separado por comas en lista de floats.

    Ejemplo:
        "0.5,1,2,4" -> [0.5, 1.0, 2.0, 4.0]
    """
    values = []

    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))

    if len(values) == 0:
        raise ValueError("La lista de factores de escala no puede estar vacía.")

    return values


def parse_int_list(text: str) -> list[int]:
    """
    Convierte un string separado por comas en lista de enteros.

    Ejemplo:
        "20,40,60,80" -> [20, 40, 60, 80]
    """
    values = []

    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))

    if len(values) == 0:
        raise ValueError("La lista de factores de calidad no puede estar vacía.")

    return values


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Construye el parser de argumentos de línea de comandos.

    Returns
    -------
    argparse.ArgumentParser
        Parser configurado.
    """
    parser = argparse.ArgumentParser(
        description="Compresor de imágenes basado en DCT."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Ruta de la imagen de entrada."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Carpeta donde se guardarán los resultados."
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Factor de escala para la matriz de cuantización."
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=None,
        help="Factor de calidad tipo JPEG entre 1 y 100. Si se usa, ignora --scale."
    )

    parser.add_argument(
        "--sweep-scales",
        type=str,
        default=None,
        help="Lista de factores de escala separados por coma. Ejemplo: 0.25,0.5,1,2,4"
    )

    parser.add_argument(
        "--sweep-qualities",
        type=str,
        default=None,
        help="Lista de calidades separadas por coma. Ejemplo: 20,40,60,80,95"
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Desactiva la generación de gráficos."
    )

    return parser


def main() -> None:
    """
    Función principal.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Caso 1: barrido de escalas
    # ============================================================
    if args.sweep_scales is not None:
        scales = parse_float_list(args.sweep_scales)

        results = run_scale_sweep(
            image_path=image_path,
            output_dir=output_dir,
            scales=scales,
        )

        csv_path = output_dir / "results_sweep_scales.csv"
        save_results_csv(results, csv_path)

        if not args.no_plots:
            save_plots(results, output_dir)

        print("=" * 60)
        print(f"Resultados guardados en: {csv_path}")
        print("=" * 60)
        return

    # ============================================================
    # Caso 2: barrido de factores de calidad
    # ============================================================
    if args.sweep_qualities is not None:
        qualities = parse_int_list(args.sweep_qualities)

        results = run_quality_sweep(
            image_path=image_path,
            output_dir=output_dir,
            qualities=qualities,
        )

        csv_path = output_dir / "results_sweep_qualities.csv"
        save_results_csv(results, csv_path)

        if not args.no_plots:
            save_plots(results, output_dir)

        print("=" * 60)
        print(f"Resultados guardados en: {csv_path}")
        print("=" * 60)
        return

    # ============================================================
    # Caso 3: ejecución única
    # ============================================================
    results = run_compression(
        image_path=image_path,
        output_dir=output_dir,
        scale=args.scale,
        quality=args.quality,
        save_image=True,
    )

    csv_path = output_dir / "results_single_run.csv"
    save_results_csv([results], csv_path)

    print_summary_from_results(results)
    print()
    print(f"Resultados guardados en: {csv_path}")


if __name__ == "__main__":
    main()