from preprocessing import preprocess_image, load_image, rgb_to_grayscale
from dct import create_dct_matrix, dct2_blocks
from quantization import (
    get_luminance_quantization_matrix,
    scale_quantization_matrix,
    quantize_blocks
)
from coding import encode_quantized_blocks
from reconstruction import reconstruct_from_encoded_blocks, save_grayscale_image
from metrics import (
    compute_quality_metrics,
    compute_compression_metrics,
    print_quality_metrics,
    print_compression_metrics
)

image_path = "imagen_prueba.png"

original_rgb = load_image(image_path)
original_gray = rgb_to_grayscale(original_rgb)

blocks, info = preprocess_image(image_path)

C = create_dct_matrix(info.block_size)
coeff_blocks = dct2_blocks(blocks, dct_matrix=C)

q_base = get_luminance_quantization_matrix()
q_matrix = scale_quantization_matrix(q_base, scale=1.0)

quantized_blocks = quantize_blocks(coeff_blocks, q_matrix)
encoded_blocks = encode_quantized_blocks(quantized_blocks)

reconstructed_image = reconstruct_from_encoded_blocks(
    encoded_blocks=encoded_blocks,
    blocks_shape=quantized_blocks.shape,
    q_matrix=q_matrix,
    info=info,
    dct_matrix=C
)

save_grayscale_image(reconstructed_image, "reconstructed_image.png")

quality = compute_quality_metrics(original_gray, reconstructed_image)

compression = compute_compression_metrics(
    quantized_blocks=quantized_blocks,
    encoded_blocks=encoded_blocks,
    original_image_shape=info.original_shape
)

print_quality_metrics(quality)
print()
print_compression_metrics(compression)