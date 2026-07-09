from preprocessing import load_image
from dct import create_dct_matrix
from quantization import get_color_quantization_matrices, print_block_class_summary
from color import (
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    split_ycbcr_channels,
    merge_ycbcr_channels,
    compress_color_channel,
    reconstruct_color_channel,
    compress_color_channel_adaptive,
    reconstruct_color_channel_adaptive,
    downsample_chrominance_420,
    upsample_chrominance_420,
    save_rgb_image,
)
from coding import huffman_encode_encoded_blocks, print_huffman_summary

# ============================================================
# Parámetros
# ============================================================

image_path = "imagen_prueba_color.png"
output_path = "reconstructed_color.png"

block_size = 8
scale = 4.0

# ============================================================
# Carga y conversión de color
# ============================================================

original_rgb = load_image(image_path)

original_ycbcr = rgb_to_ycbcr(original_rgb)
y, cb, cr = split_ycbcr_channels(original_ycbcr)

cb_down, cr_down, chroma_info = downsample_chrominance_420(cb, cr)

# ============================================================
# Matrices DCT y cuantización
# ============================================================

C = create_dct_matrix(block_size)

q_y, q_cb, q_cr = get_color_quantization_matrices(
    scale=scale
)

# ============================================================
# Compresión por componente
# ============================================================

encoded_y, quantized_y, info_y, class_map_y, scale_map_y = compress_color_channel_adaptive(
    y,
    q_matrix=q_y,
    dct_matrix=C,
    block_size=block_size,
    smooth_scale=1.5,
    edge_scale=0.75,
    texture_scale=1.0
)

print()
print_block_class_summary(class_map_y)

encoded_cb, quantized_cb, info_cb = compress_color_channel(
    cb_down,
    q_matrix=q_cb,
    dct_matrix=C,
    block_size=block_size
)

encoded_cr, quantized_cr, info_cr = compress_color_channel(
    cr_down,
    q_matrix=q_cr,
    dct_matrix=C,
    block_size=block_size
)

huffman_y = huffman_encode_encoded_blocks(encoded_y)
huffman_cb = huffman_encode_encoded_blocks(encoded_cb)
huffman_cr = huffman_encode_encoded_blocks(encoded_cr)

print()
print("Huffman Y")
print_huffman_summary(huffman_y)

print()
print("Huffman Cb")
print_huffman_summary(huffman_cb)

print()
print("Huffman Cr")
print_huffman_summary(huffman_cr)

# ============================================================
# Reconstrucción por componente
# ============================================================

reconstructed_y = reconstruct_color_channel_adaptive(
    encoded_y,
    blocks_shape=quantized_y.shape,
    q_matrix=q_y,
    info=info_y,
    dct_matrix=C,
    scale_map=scale_map_y
)

reconstructed_cb_down = reconstruct_color_channel(
    encoded_cb,
    blocks_shape=quantized_cb.shape,
    q_matrix=q_cb,
    info=info_cb,
    dct_matrix=C
)

reconstructed_cr_down = reconstruct_color_channel(
    encoded_cr,
    blocks_shape=quantized_cr.shape,
    q_matrix=q_cr,
    info=info_cr,
    dct_matrix=C
)

# ============================================================
# Unión YCbCr y conversión a RGB
# ============================================================

reconstructed_cb, reconstructed_cr = upsample_chrominance_420(
    reconstructed_cb_down,
    reconstructed_cr_down,
    chroma_info
)

reconstructed_ycbcr = merge_ycbcr_channels(
    reconstructed_y,
    reconstructed_cb,
    reconstructed_cr
)

reconstructed_rgb = ycbcr_to_rgb(reconstructed_ycbcr)

save_rgb_image(reconstructed_rgb, output_path)

print("Compresión color completada.")
print(f"Imagen original RGB     : {original_rgb.shape}")
print(f"Imagen reconstruida RGB : {reconstructed_rgb.shape}")
print(f"Imagen guardada en      : {output_path}")