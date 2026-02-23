from __future__ import annotations

from typing import Sequence

import os
import numpy as np
import matplotlib.pyplot as plt

"""
    K-nearest neighbors (KNN) classification example
    
    Creates a dataset with classified training points (categories 1, 2, 3, 4, etc.) and
    a dataset of unclassified test points that need to be classified using the KNN algorithm

    KNN uses absolute distance between points, meaning that it only works if numerical distance between 
    points also makes sense, where a shorter distance means a more similar point

    Author :        Martijn Folmer
    Date created :  04-02-2026
"""


def _euclidean_distance(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between two sets of points.
    Returns shape (n_points1, n_points2) where entry [i, j] is the distance
    between points1[i] and points2[j].
    """
    diffs = points1[:, None, :] - points2[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))


def knn_classify(
    training_points: Sequence[Sequence[float]],
    training_labels: Sequence[int],
    query_points: Sequence[Sequence[float]],
    k: int
) -> np.ndarray:
    """
    Classify query points using K-nearest neighbors algorithm.
    
    Args:
        training_points: Sequence of training points, shape (n_train, n_dims)
        training_labels: Sequence of category labels for training points, shape (n_train,)
        query_points: Sequence of points to classify, shape (n_query, n_dims)
        k: Number of nearest neighbors to consider
    
    Returns:
        Predicted labels for query points, shape (n_query,)
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    if len(training_points) != len(training_labels):
        raise ValueError("training_points and training_labels must have the same length")
    if k > len(training_points):
        raise ValueError(f"k ({k}) cannot be larger than number of training points ({len(training_points)})")
    
    training_array = np.asarray(training_points, dtype=float)
    query_array = np.asarray(query_points, dtype=float)
    labels_array = np.asarray(training_labels, dtype=int)
    
    if training_array.ndim != 2 or query_array.ndim != 2:
        raise ValueError("points must be 2D arrays of shape (n_points, n_dims)")
    if training_array.shape[1] != query_array.shape[1]:
        raise ValueError("training_points and query_points must have the same number of dimensions")
    
    # Compute distances from each query point to all training points
    distances = _euclidean_distance(query_array, training_array)
    
    # For each query point, find the k nearest neighbors
    predicted_labels = []
    for i in range(len(query_points)):
        # Get indices of k nearest neighbors
        nearest_indices = np.argsort(distances[i])[:k]
        # Get labels of k nearest neighbors
        nearest_labels = labels_array[nearest_indices]
        # Majority vote (most common label)
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        predicted_labels.append(predicted_label)
    
    return np.array(predicted_labels)


def plot_knn_results(
    training_points: Sequence[Sequence[float]],
    training_labels: Sequence[int],
    test_points: Sequence[Sequence[float]],
    predicted_labels: Sequence[int],
    k_value: int,
    title: str = "K-Nearest Neighbors Classification"
) -> None:
    """
    Plot KNN classification results showing training points (by true category) with circles and
    test points (by predicted category) with triangles.
    """
    training_array = np.asarray(training_points, dtype=float)
    test_array = np.asarray(test_points, dtype=float)
    training_labels_array = np.asarray(training_labels, dtype=int)
    predicted_labels_array = np.asarray(predicted_labels, dtype=int)
    
    if training_array.shape[1] != 2:
        raise ValueError("This plotting function only supports 2D points")
    
    # Get unique categories
    all_categories = sorted(set(training_labels_array.tolist() + predicted_labels_array.tolist()))
    
    # Create color map for categories
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_categories)))
    category_to_color = {cat: colors[i] for i, cat in enumerate(all_categories)}
    
    # Plot training points (circles) colored by true category
    for category in all_categories:
        mask = training_labels_array == category
        if np.any(mask):
            plt.scatter(
                training_array[mask, 0],
                training_array[mask, 1],
                c=[category_to_color[category]],
                marker='o',
                s=50,
                alpha=0.7,
                label=f'Training Category {category}',
                edgecolors='black',
                linewidths=0.5
            )
    
    # Plot test points (triangles) colored by predicted category
    for category in all_categories:
        mask = predicted_labels_array == category
        if np.any(mask):
            plt.scatter(
                test_array[mask, 0],
                test_array[mask, 1],
                c=[category_to_color[category]],
                marker='^',
                s=80,
                alpha=0.8,
                label=f'Test Predicted {category}',
                edgecolors='black',
                linewidths=0.5
            )
    
    plt.title(f"{title} (k={k_value})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


if __name__ == "__main__":
    
    # Set random seed for reproducibility
    rng = np.random.default_rng(42)
    
    # Parameters for generating classified training data
    n_categories = 4
    pts_per_category = 50
    category_spread = 0.3  # smaller = tighter clusters
    
    # Parameters for generating unclassified test data
    n_test_points = 30
    
    # Parameters for KNN
    k_value = 5
    
    ####################
    # Generate classified training dataset
    ####################
    
    # Create centers for each category
    centers = rng.uniform(low=[-3, -3], high=[8, 8], size=(n_categories, 2))
    
    training_points_list = []
    training_labels_list = []
    
    for category_id in range(1, n_categories + 1):
        center = centers[category_id - 1]
        # Generate points around this center using normal distribution
        points = rng.normal(loc=center, scale=category_spread, size=(pts_per_category, 2))
        training_points_list.append(points)
        training_labels_list.extend([category_id] * pts_per_category)
    
    training_points = np.vstack(training_points_list)
    training_labels = np.array(training_labels_list)
    
    print(f"Generated {len(training_points)} training points across {n_categories} categories")
    print(f"Category distribution: {dict(zip(*np.unique(training_labels, return_counts=True)))}")
    
    ####################
    # Generate unclassified test dataset
    ####################
    
    # Generate test points distributed randomly across the same space
    test_points = rng.uniform(low=[-2, -2], high=[7, 7], size=(n_test_points, 2))
    
    print(f"\nGenerated {len(test_points)} test points to classify")
    
    ####################
    # Classify test points using KNN
    ####################
    
    predicted_labels = knn_classify(
        training_points=training_points,
        training_labels=training_labels,
        query_points=test_points,
        k=k_value
    )
    
    print(f"\nClassification results (k={k_value}):")
    print(f"Predicted category distribution: {dict(zip(*np.unique(predicted_labels, return_counts=True)))}")
    
    ####################
    # Visualize and save results
    ####################
    
    plot_knn_results(
        training_points=training_points,
        training_labels=training_labels,
        test_points=test_points,
        predicted_labels=predicted_labels,
        k_value=k_value,
        title="K-Nearest Neighbors Classification"
    )
    
    # Save the plot
    output_dir = "readme_img"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "knnClassification.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Show the plot
    plt.show()

