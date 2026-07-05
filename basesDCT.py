import numpy as np
import matplotlib.pyplot as plt

# ======================================
# Parámetros
# ======================================
N = 8  # tamaño del bloque DCT

# ======================================
# Factor de normalización
# ======================================
def alpha(k, N):
    return np.sqrt(1/N) if k == 0 else np.sqrt(2/N)

# ======================================
# Función base 2D de la DCT
# ======================================
def dct_basis(u, v, N):
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing='ij')

    basis = (
        alpha(u, N)
        * alpha(v, N)
        * np.cos((np.pi / N) * (X + 0.5) * u)
        * np.cos((np.pi / N) * (Y + 0.5) * v)
    )
    return basis

# ======================================
# Función para graficar un grupo 4x4
# u_range y v_range indican qué bases mostrar
# ======================================
def plot_dct_group(u_range, v_range, filename, title):
    fig, axes = plt.subplots(len(u_range), len(v_range), figsize=(10, 10))

    for i, u in enumerate(u_range):
        for j, v in enumerate(v_range):
            ax = axes[i, j]
            basis = dct_basis(u, v, N)

            ax.imshow(
                basis,
                cmap="gray",
                interpolation="nearest",
                vmin=-0.25,
                vmax=0.25
            )

            ax.set_title(f"$({u},{v})$", fontsize=12, pad=4)
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(f"{filename}.pdf", bbox_inches="tight")
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.show()

# ======================================
# Generar 4 figuras separadas
# ======================================

# Figura 1: bajas frecuencias
plot_dct_group(
    u_range=range(0, 4),
    v_range=range(0, 4),
    filename="dct_bases_0_3_0_3",
    title="Funciones base de la DCT 2D ($u=0\\ldots3$, $v=0\\ldots3$)"
)

# Figura 2
plot_dct_group(
    u_range=range(0, 4),
    v_range=range(4, 8),
    filename="dct_bases_0_3_4_7",
    title="Funciones base de la DCT 2D ($u=0\\ldots3$, $v=4\\ldots7$)"
)

# Figura 3
plot_dct_group(
    u_range=range(4, 8),
    v_range=range(0, 4),
    filename="dct_bases_4_7_0_3",
    title="Funciones base de la DCT 2D ($u=4\\ldots7$, $v=0\\ldots3$)"
)

# Figura 4: altas frecuencias
plot_dct_group(
    u_range=range(4, 8),
    v_range=range(4, 8),
    filename="dct_bases_4_7_4_7",
    title="Funciones base de la DCT 2D ($u=4\\ldots7$, $v=4\\ldots7$)"
)