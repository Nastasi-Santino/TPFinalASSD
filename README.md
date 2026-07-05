# TPFinalASSD — Compresor de imágenes tipo JPEG

Trabajo Práctico N°4 — 25.20 Análisis de Señales y Sistemas Digitales (ITBA)
Grupo 1: Nastasi Santino, Porco Paneiva Paloma, Amorbello Renzo, Cecotti Fabrizio

Implementación modular de un compresor de imágenes con pérdida, inspirado en el
esquema JPEG, basado en la Transformada Discreta del Coseno (DCT/TDC). El
sistema divide la imagen en bloques de 8×8, aplica DCT 2D, cuantiza los
coeficientes con una matriz configurable, los codifica simbólicamente
(zig-zag + DC diferencial + RLE) y reconstruye la imagen a partir de esa
representación, evaluando la calidad y la compresión obtenidas.

> Este README fue armado a partir del informe (`ASSDTP4_Grupo1.pdf`) y del
> código fuente, ya que el repo no tenía documentación propia.

---

## Índice

1. [Pipeline general](#pipeline-general)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Instalación](#instalación)
4. [Uso rápido](#uso-rápido)
5. [Uso desde línea de comandos (`main.py`)](#uso-desde-línea-de-comandos-mainpy)
6. [Descripción de cada módulo](#descripción-de-cada-módulo)
7. [Métricas que se calculan](#métricas-que-se-calculan)
8. [Notas conceptuales clave](#notas-conceptuales-clave)
9. [Limitaciones actuales / posibles extensiones](#limitaciones-actuales--posibles-extensiones)

---

## Pipeline general

**Compresión:**

```
Imagen original (RGB)
  → Escala de grises
  → Padding por replicación de borde (múltiplos de 8)
  → División en bloques de 8x8
  → Corrimiento de nivel (-128)
  → DCT 2D por bloque
  → Cuantización (matriz Q escalada o por factor de calidad)
  → Codificación: zig-zag + DC diferencial + RLE de AC
  → EncodedBlock[] (representación simbólica, "datos comprimidos")
```

**Reconstrucción (inversa):**

```
EncodedBlock[]
  → Decodificación RLE + DC diferencial
  → Reordenamiento zig-zag inverso
  → De-cuantización (multiplicar por Q)
  → IDCT 2D por bloque
  → Corrimiento de nivel inverso (+128)
  → Reensamblado de bloques
  → Crop al tamaño original
  → Clipping [0,255] + conversión a uint8
  → Imagen reconstruida (escala de grises)
```

Sobre esta comparación (original vs. reconstruida) se calculan métricas de
calidad (MSE, MAE, PSNR) y de compresión (porcentaje de ceros, símbolos RLE,
bits estimados, relación de compresión).

> **Importante:** la implementación actual sólo trabaja con imágenes en
> **escala de grises**. La conversión a color (YCbCr) se menciona en el
> informe como una extensión posible, pero no está implementada.

---

## Estructura del repositorio

```
.
├── preprocessing.py     # Carga, gris, padding, bloques, level shift
├── dct.py                # Matriz DCT ortonormal, DCT/IDCT 1D y 2D
├── quantization.py       # Matriz de cuantización JPEG, escalado, quality factor
├── coding.py              # Zig-zag, DC diferencial, RLE de AC
├── reconstruction.py     # Pipeline inverso completo + guardado de imagen
├── metrics.py             # MSE, MAE, PSNR, métricas de compresión, prints
├── main.py                # CLI: ejecución única o barridos (scale/quality), CSV, plots
├── simple_main.py         # Script mínimo de ejemplo (sin CLI, un solo run)
├── basesDCT.py            # Script standalone: grafica las 64 funciones base 8x8 de la DCT
├── imagen_prueba.png      # Imagen de entrada usada por los scripts de ejemplo
├── ASSDTP4_Grupo1.pdf     # Informe teórico del TP
└── README.md
```

Cada módulo tiene, además, un bloque `if __name__ == "__main__":` con un
ejemplo mínimo de uso / autoverificación (correr `python <modulo>.py` prueba
ese módulo de forma aislada).

---

## Instalación

Dependencias: `numpy`, `Pillow` (PIL), y opcionalmente `matplotlib` (para
`main.py --sweep-*` y para `basesDCT.py`).

```bash
pip install numpy pillow matplotlib
```

---

## Uso rápido

### Opción 1: script mínimo (`simple_main.py`)

Corre todo el pipeline sobre `imagen_prueba.png` con `scale=1.0`, guarda
`reconstructed_image.png` e imprime las métricas. Es el punto de entrada más
simple para entender el flujo:

```bash
python simple_main.py
```

### Opción 2: como librería

```python
from preprocessing import preprocess_image, load_image, rgb_to_grayscale
from dct import create_dct_matrix, dct2_blocks
from quantization import get_luminance_quantization_matrix, scale_quantization_matrix, quantize_blocks
from coding import encode_quantized_blocks
from reconstruction import reconstruct_from_encoded_blocks, save_grayscale_image
from metrics import compute_quality_metrics, compute_compression_metrics, print_quality_metrics, print_compression_metrics

image_path = "imagen_prueba.png"

original_gray = rgb_to_grayscale(load_image(image_path))
blocks, info = preprocess_image(image_path)

C = create_dct_matrix(info.block_size)
coeff_blocks = dct2_blocks(blocks, dct_matrix=C)

q_matrix = scale_quantization_matrix(get_luminance_quantization_matrix(), scale=1.0)
quantized_blocks = quantize_blocks(coeff_blocks, q_matrix)

encoded_blocks = encode_quantized_blocks(quantized_blocks)

reconstructed = reconstruct_from_encoded_blocks(
    encoded_blocks=encoded_blocks,
    blocks_shape=quantized_blocks.shape,
    q_matrix=q_matrix,
    info=info,
    dct_matrix=C,
)
save_grayscale_image(reconstructed, "reconstructed_image.png")

quality = compute_quality_metrics(original_gray, reconstructed)
compression = compute_compression_metrics(quantized_blocks, encoded_blocks, info.original_shape)
print_quality_metrics(quality)
print_compression_metrics(compression)
```

---

## Uso desde línea de comandos (`main.py`)

`main.py` es la interfaz "completa": corre el pipeline, guarda la imagen
reconstruida, calcula métricas, las exporta a CSV y (opcionalmente) genera
gráficos. Soporta tres modos:

### 1. Ejecución única

```bash
python main.py --image imagen_prueba.png --output-dir results --scale 1.0
```

o usando factor de calidad tipo JPEG (1–100) en lugar de `scale`:

```bash
python main.py --image imagen_prueba.png --output-dir results --quality 80
```

Genera `results/reconstructed_scale_1.png` (o `_quality_80.png`) y
`results/results_single_run.csv`.

### 2. Barrido de factores de escala

```bash
python main.py --image imagen_prueba.png --output-dir results --sweep-scales 0.25,0.5,1,2,4
```

Corre la compresión para cada valor de `scale`, guarda una imagen
reconstruida por cada uno, un CSV consolidado
(`results_sweep_scales.csv`) y tres gráficos (PSNR, relación de compresión y
% de ceros en función del scale), salvo que se pase `--no-plots`.

### 3. Barrido de factores de calidad

```bash
python main.py --image imagen_prueba.png --output-dir results --sweep-qualities 20,40,60,80,95
```

Análogo al anterior pero variando `quality` (1–100, estilo JPEG) en vez de
`scale`.

### Argumentos disponibles

| Argumento | Descripción |
|---|---|
| `--image` | Ruta de la imagen de entrada (**requerido**) |
| `--output-dir` | Carpeta de salida (default `results`) |
| `--scale` | Factor de escala de la matriz de cuantización (default `1.0`) |
| `--quality` | Factor de calidad JPEG 1–100 (si se usa, ignora `--scale`) |
| `--sweep-scales` | Lista de scales separados por coma, ej. `0.25,0.5,1,2,4` |
| `--sweep-qualities` | Lista de qualities separados por coma, ej. `20,40,60,80,95` |
| `--no-plots` | Desactiva la generación de gráficos matplotlib |

---

## Descripción de cada módulo

### `preprocessing.py`
Prepara la imagen antes de la DCT.
- `load_image(path)` → carga como RGB uint8.
- `rgb_to_grayscale(img)` → luminancia `Y = 0.299R + 0.587G + 0.114B`.
- `compute_padding` / `pad_image_edge` → rellena por **replicación de borde**
  hasta que alto/ancho sean múltiplos de 8 (no usa ceros, para evitar
  discontinuidades artificiales).
- `split_into_blocks` → divide en bloques no solapados `(nv, nh, 8, 8)`.
- `level_shift_blocks` → resta 128 para centrar en `[-128,127]`.
- `preprocess_image(path)` → corre todo lo anterior y devuelve
  `(shifted_blocks, PreprocessingInfo)`.
- `PreprocessingInfo`: guarda `original_shape`, `padded_shape`, `pad_height`,
  `pad_width`, `block_size` — necesario para poder reconstruir y recortar
  correctamente después.

### `dct.py`
DCT ortonormal tipo II, implementada "a mano" (sin `scipy.fftpack`).
- `create_dct_matrix(N)` → matriz `C` de `N×N` (por defecto `N=8`).
- `is_orthonormal(C)` → chequeo de que `C @ C.T == I`.
- `dct_1d` / `idct_1d` → transformada 1D (`X = C@x`, `x = C.T@X`).
- `dct2_block` / `idct2_block` → 2D vía forma separable: `F = C @ B @ C.T`.
- `dct2_blocks` / `idct2_blocks` → aplica lo anterior a todo el arreglo de
  bloques `(nv, nh, N, N)`.

### `quantization.py`
Cuantización escalar dependiente de la frecuencia.
- `JPEG_LUMINANCE_QUANTIZATION_MATRIX` → matriz `Q_Y` estándar de JPEG (8×8).
- `scale_quantization_matrix(Q, scale)` → `Q_s = round(scale * Q)`, clipeada
  a `[1,255]` para evitar pasos nulos.
- `quality_to_scale(quality)` → convierte un quality factor JPEG (1–100) al
  `scale` equivalente (fórmula estándar: `5000/quality` si `<50`,
  `200-2*quality` si `>=50`).
- `create_quality_quantization_matrix(quality)` → atajo que combina las dos
  anteriores.
- `quantize_block(s)` / `dequantize_block(s)` → `round(F/Q)` y `F̂·Q`
  respectivamente, en versión bloque único y en batch (`_blocks`).
- `count_zero_coefficients` / `zero_percentage` → métricas rápidas sobre
  cuántos coeficientes se anularon.

### `coding.py`
Reordenamiento y codificación simbólica (sin Huffman real).
- `create_zigzag_indices(N)` / `zigzag_scan` / `inverse_zigzag_scan` →
  recorrido zig-zag que ordena de baja a alta frecuencia.
- `split_dc_ac` → separa el primer coeficiente (DC) del resto (AC, 63 valores
  para N=8).
- `differential_encode_dc` / `differential_decode_dc` → codifica el DC como
  diferencia respecto del bloque anterior (`ΔDC_n = DC_n - DC_{n-1}`).
- `rle_encode_ac` / `rle_decode_ac` → codifica los AC como pares
  `(run_length, amplitude)` + símbolo `"EOB"` cuando el resto son ceros.
- `EncodedBlock` (dataclass) → `dc_diff: int`, `ac_symbols: list[(int,int)|str]`.
  Es la "unidad" de datos comprimidos que viaja entre `coding.py` y
  `reconstruction.py`.
- `encode_quantized_blocks` / `decode_quantized_blocks` → aplican todo lo
  anterior a un arreglo completo de bloques, en orden raster scan.
- `count_rle_symbols`, `average_rle_symbols_per_block`, `count_eob_symbols` →
  estadísticas usadas por `metrics.py`.

### `reconstruction.py`
Pipeline inverso completo, hasta obtener una imagen `uint8` guardable.
- `inverse_level_shift_blocks` → `+128`.
- `merge_blocks` → reensambla bloques `(nv, nh, N, N)` en una imagen 2D.
- `crop_to_original_shape` → recorta el padding agregado en el preprocesamiento.
- `clip_image` / `image_to_uint8` → satura a `[0,255]` y castea a `uint8`.
- `save_grayscale_image` → guarda con PIL en modo `"L"`.
- `reconstruct_from_blocks` → desde bloques espaciales ya de-cuantizados+IDCT.
- `reconstruct_from_quantized_blocks` → agrega de-cuantización + IDCT.
- `reconstruct_from_encoded_blocks` → **función de entrada típica**: parte de
  `list[EncodedBlock]` y hace decodificación + de-cuantización + IDCT +
  reensamblado + crop + clipping, todo en un solo llamado.

### `metrics.py`
Evaluación cuantitativa de calidad y compresión (no realiza compresión ni
reconstrucción por sí mismo).
- **Calidad**: `mse`, `mae`, `psnr` → agrupadas en `compute_quality_metrics`
  (dataclass `QualityMetrics`).
- **Compresión** (dataclass `CompressionMetrics`, vía
  `compute_compression_metrics`):
  - `count_total/zero/nonzero_coefficients`, `zero_percentage`.
  - `rle_symbols`, `average_rle_symbols_per_block` (de `coding.py`).
  - `estimate_original_bits` = `alto*ancho*8`.
  - `estimate_bits_from_rle` — modelo simple: `dc_diff` a 16 bits fijos por
    bloque, cada símbolo `(run,amp)` a `6+16` bits, cada `EOB` a 4 bits.
    (Es una **estimación**, no un bitstream real ni codificación entrópica).
  - `compression_ratio` = `original_bits/compressed_bits`.
  - `reduction_percentage` = `(1 - compressed/original) * 100`.
- `print_quality_metrics` / `print_compression_metrics` → salida prolija por
  consola.

### `main.py`
Orquestador de alto nivel + CLI (ver sección de uso más arriba).
- `run_compression(...)` → corre el pipeline completo para una imagen y una
  configuración de cuantización (por `scale` o por `quality`), devuelve un
  `dict` con todas las métricas + metadatos de la corrida.
- `run_scale_sweep` / `run_quality_sweep` → repiten `run_compression` para
  una lista de valores.
- `save_results_csv` → exporta resultados a CSV.
- `save_plots` → grafica PSNR, relación de compresión y % de ceros vs.
  scale/quality (usa matplotlib si está disponible).
- `print_summary_from_results` → resumen legible por consola.

### `simple_main.py`
Versión mínima sin argparse, útil para pruebas rápidas y para leer el
pipeline "de punta a punta" sin la capa de CLI de `main.py`.

### `basesDCT.py`
Script **standalone** (no se importa desde el resto del proyecto) que grafica
las 64 funciones base 2D de la DCT de 8×8, agrupadas en 4 figuras de 4×4
(bajas/altas frecuencias combinadas), tal como se muestra en las Figuras 1–4
del informe. Genera `.png` y `.pdf` con `matplotlib`.

---

## Métricas que se calculan

**Calidad de reconstrucción** (comparando imagen original vs. reconstruida):
- **MSE** (error cuadrático medio)
- **MAE** (error absoluto medio)
- **PSNR** (relación señal a ruido pico, en dB; `∞` si MSE=0)

**Eficiencia de compresión:**
- % de coeficientes cuantizados iguales a cero
- Cantidad total/promedio de símbolos RLE por bloque
- Bits estimados de la imagen original vs. la representación comprimida
- Relación de compresión (`CR`) y porcentaje de reducción

> El informe también menciona **SSIM** (sección 7.7) como métrica
> complementaria de similitud estructural, pero **no está implementada** en
> el código (`metrics.py` no tiene función SSIM).

---

## Notas conceptuales clave (resumen del informe)

- La **DCT por sí sola no comprime ni pierde información**: es una
  transformación ortonormal e invertible exactamente si se preservan los
  coeficientes con precisión infinita. La única fuente de pérdida real es la
  **cuantización**.
- Los bloques son de **8×8** (igual que JPEG base): compromiso entre
  compactación de energía, costo computacional y localización espacial.
- El **corrimiento de nivel** (`-128`) centra los valores de píxel
  `[0,255] → [-128,127]` antes de la DCT, lo cual ayuda a que el coeficiente
  DC represente mejor el nivel medio del bloque.
- La matriz de cuantización `Q_Y` es más "permisiva" (valores chicos) en
  bajas frecuencias (esquina superior izquierda) y más agresiva (valores
  grandes) en altas frecuencias, aprovechando que el ojo humano es menos
  sensible a estas últimas.
- El **factor de escala `s`** (`Q_s = s·Q_Y`) controla el compromiso
  calidad/compresión: `s<1` = cuantización más suave (mejor PSNR, peor
  compresión); `s>1` = más agresiva (peor PSNR, mejor compresión). El
  `quality` (1–100) es una forma alternativa de parametrizar lo mismo, estilo
  JPEG.
- El recorrido **zig-zag** reordena los coeficientes de baja a alta
  frecuencia para agrupar los ceros (generados por la cuantización) al final
  del vector, lo que hace eficiente el **RLE**.
- El **DC se codifica diferencialmente** entre bloques consecutivos porque
  suele variar lentamente (bloques vecinos con niveles medios similares).
- Los **artefactos de bloque** (discontinuidades visibles en la grilla de
  8×8) aparecen cuando la cuantización es muy agresiva, porque cada bloque
  se procesa de forma independiente.
- El proyecto usa **RLE pero no Huffman/codificación entrópica real**: las
  métricas de bits/compresión son **estimaciones** basadas en un modelo fijo
  de bits por símbolo (ver `estimate_bits_from_rle` en `metrics.py`), no un
  bitstream real.

---

## Limitaciones actuales / posibles extensiones

Según lo planteado en el informe como trabajo futuro, y confirmado en el
código:

- ❌ Sin soporte de **imágenes a color** (RGB→YCbCr con tratamiento separado
  de luminancia/crominancia). Todo se procesa en escala de grises.
- ❌ Sin **codificación entrópica real** (Huffman). El "compressed_bits" es
  una estimación con bits fijos por tipo de símbolo, no un bitstream.
- ❌ Sin **SSIM** implementado (mencionado en el informe, sección 7.7).
- ✅ Sí implementado: DCT/IDCT propia, cuantización con matriz JPEG y factor
  de escala/calidad, zig-zag, DC diferencial, RLE con EOB, reconstrucción
  completa, métricas MSE/MAE/PSNR y de compresión, CLI con barridos y
  generación de CSV/gráficos.

---

## Referencias (del informe)

1. N. Ahmed, T. Natarajan, K. R. Rao, "Discrete Cosine Transform," *IEEE
   Trans. Computers*, 1974.
2. K. R. Rao, P. Yip, *Discrete Cosine Transform: Algorithms, Advantages,
   Applications*, Academic Press, 1990.
3. G. K. Wallace, "The JPEG Still Picture Compression Standard," *IEEE
   Trans. Consumer Electronics*, 1992.
4. CCITT, Recommendation T.81, 1992.
5. A. M. Raid et al., "Jpeg Image Compression Using Discrete Cosine
   Transform – A Survey," 2014.
6–10. Ver PDF del informe (`ASSDTP4_Grupo1.pdf`) para el resto de las
   referencias, incluyendo SSIM (Wang et al., 2004).