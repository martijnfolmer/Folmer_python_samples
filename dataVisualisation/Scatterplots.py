import matplotlib.pyplot as plt
import numpy as np

"""
   Plot 2D and 3D scatterplots with random colors and sizes, with the basic features of x/y labels, cmapping, grid,
   tight_layout, etc.

    Author :        Martijn Folmer
    Date created :  29-06-2026
"""

def plot_2d_scatter(x, y, colors, sizes):
    """Create a 2D scatter plot"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create scatter plot (c=color array, s=size array, cmap=color map)
    scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.7, cmap='viridis', edgecolors='w')

    # Formatting
    ax.set_title("2D Scatter Plot Example", fontsize=14, fontweight='bold')
    ax.set_xlabel("X-axis Label", fontsize=12)
    ax.set_ylabel("Y-axis Label", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Color Intensity", fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_3d_scatter(x, y, z, colors, sizes):
    """Create 3D scatter plot"""
    fig = plt.figure(figsize=(10, 8))

    # projection='3d' enables the 3D axes, allows moving the graph around
    ax = fig.add_subplot(111, projection='3d')

    # Create 3D scatter plot
    scatter = ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.7, cmap='plasma', edgecolors='k', linewidths=0.5)

    # Formatting
    ax.set_title("3D Scatter Plot Example", fontsize=14, fontweight='bold')
    ax.set_xlabel("X-axis Label", fontsize=12)
    ax.set_ylabel("Y-axis Label", fontsize=12)
    ax.set_zlabel("Z-axis Label", fontsize=12)

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Color Intensity", fontsize=10)

    plt.tight_layout()
    plt.show()


# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    # dummy data
    np.random.seed(42)
    num_points = 150

    x = np.random.rand(num_points)
    y = np.random.rand(num_points)
    z = np.random.rand(num_points)

    colors = np.random.rand(num_points)
    sizes = 500 * np.random.rand(num_points)


    print("Generating 2D Scatterplot:")
    plot_2d_scatter(x, y, colors, sizes)

    print("Generating 3D Scatterplot:")
    plot_3d_scatter(x, y, z, colors, sizes)