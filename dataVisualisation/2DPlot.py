import matplotlib.pyplot as plt

"""
   Plot 2D graphs in different ways using matplotlib

    Included plots:
    - Line graph
    - Scatter plot
    - Bar graph
    - Pie chart

    Author :        Martijn Folmer
    Date created :  23-01-2026
"""


def plot_comparison_graphs():
    # Sample data
    x = [1, 2, 3, 4, 5]

    y1 = [2, 4, 6, 8, 10]
    y2 = [1, 3, 5, 7, 9]

    pie_data = [40, 35, 25]
    labels = ["Category A", "Category B", "Category C"]

    bar_width = 0.35
    x_indices = range(len(x))

    # Create 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # -------------------- Line Plot --------------------
    axs[0, 0].plot(x, y1, label="Dataset A")
    axs[0, 0].plot(x, y2, label="Dataset B")
    axs[0, 0].set_title("Line Graph Comparison")
    axs[0, 0].set_xlabel("X values")
    axs[0, 0].set_ylabel("Y values")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].set_xlim(0, 6)
    axs[0, 0].set_ylim(0, 12)

    # -------------------- Scatter Plot --------------------
    axs[0, 1].scatter(x, y1, label="Dataset A")
    axs[0, 1].scatter(x, y2, label="Dataset B")
    axs[0, 1].set_title("Scatter Plot Comparison")
    axs[0, 1].set_xlabel("X values")
    axs[0, 1].set_ylabel("Y values")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[0, 1].set_xlim(0, 6)
    axs[0, 1].set_ylim(0, 12)

    # -------------------- Bar Graph --------------------
    axs[1, 0].bar(x_indices, y1, width=bar_width, label="Dataset A")
    axs[1, 0].bar(
        [i + bar_width for i in x_indices],
        y2,
        width=bar_width,
        label="Dataset B"
    )
    axs[1, 0].set_title("Bar Graph Comparison")
    axs[1, 0].set_xlabel("Category")
    axs[1, 0].set_ylabel("Value")
    axs[1, 0].legend()
    axs[1, 0].grid(True, axis="y")
    axs[1, 0].set_xlim(-0.5, len(x))
    axs[1, 0].set_ylim(0, 12)

    # -------------------- Pie Chart --------------------
    axs[1, 1].pie(
        pie_data,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )
    axs[1, 1].set_title("Pie Chart")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_comparison_graphs()
