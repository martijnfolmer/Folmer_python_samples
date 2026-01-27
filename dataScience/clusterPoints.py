from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
    Cluster points based on a simple implementation of the DBSCAN algorithm
    Note that this uses pairwise distance calculations, which is find for a small or medium dataset, but gets 
    inefficient for larger datasets

    Author :        Martijn Folmer
    Date created :  27-01-2026
"""


@dataclass(frozen=True)
class DBSCANResult:
    labels: np.ndarray          # shape (n,), cluster id per point; -1 means noise
    n_clusters: int             # number of clusters (excluding noise)
    core_sample_mask: np.ndarray  # shape (n,), True for core points


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    """
    Compute full pairwise Euclidean distances, shape (n, n).
    """
    diffs = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))


def dbscan(points: Sequence[Sequence[float]], eps: float, min_samples: int = 3) -> DBSCANResult:
    """
    Basic DBSCAN implementation (no sklearn dependency).
    - eps: neighborhood radius
    - min_samples: minimum neighbors (including the point itself) to be a core point

    Returns labels where -1 means noise.
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if min_samples <= 0:
        raise ValueError("min_samples must be >= 1")

    X = np.asarray(points, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("points must be a non-empty 2D array-like of shape (n_points, n_dims)")

    n = X.shape[0]
    dists = _pairwise_distances(X)

    neighbors = [np.where(dists[i] <= eps)[0] for i in range(n)]
    core = np.array([len(neighbors[i]) >= min_samples for i in range(n)], dtype=bool)

    UNVISITED = -2      # label for we haven't been here yet
    NOISE = -1          # label given to noise that is not part of a cluster
    labels = np.full(n, UNVISITED, dtype=int)

    cluster_id = 0

    for i in range(n):
        if labels[i] != UNVISITED:
            continue
        if not core[i]:
            labels[i] = NOISE
            continue

        # start new cluster
        labels[i] = cluster_id
        seed_queue = list(neighbors[i])

        # Expand cluster
        q = 0
        while q < len(seed_queue):
            j = seed_queue[q]
            q += 1

            if labels[j] == NOISE:
                labels[j] = cluster_id
            if labels[j] != UNVISITED:
                continue

            labels[j] = cluster_id

            if core[j]:
                # add its neighbors to the queue
                for k in neighbors[j]:
                    if k not in seed_queue:
                        seed_queue.append(k)

        cluster_id += 1

    # If there are any which we haven't classified
    labels[labels == UNVISITED] = NOISE

    return DBSCANResult(labels=labels, n_clusters=cluster_id, core_sample_mask=core)


def cluster_points_2d(points: Sequence[Point2D], eps: float, min_samples: int = 3) -> DBSCANResult:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Expected points as (n, 2)")
    return dbscan(arr, eps=eps, min_samples=min_samples)


def cluster_points_3d(points: Sequence[Point3D], eps: float, min_samples: int = 3) -> DBSCANResult:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Expected points as (n, 3)")
    return dbscan(arr, eps=eps, min_samples=min_samples)


#################
# Visualizations
#################

def plot_clusters_2d(points: Sequence[Point2D], labels: Sequence[int], title: str = "2D Clusters") -> None:
    """Plot 2D clustering result.Noise points are labeled -1 and drawn with 'x'"""

    X = np.asarray(points, dtype=float)
    labels = np.asarray(labels, dtype=int)

    unique_labels = sorted(set(labels.tolist()))
    for lab in unique_labels:
        mask = labels == lab
        if lab == -1:
            plt.scatter(X[mask, 0], X[mask, 1], marker="x")
        else:
            plt.scatter(X[mask, 0], X[mask, 1])

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()


def plot_clusters_3d(points: Sequence[Point3D], labels: Sequence[int], title: str = "3D Clusters") -> None:
    """Plot 3D clustering result. Noise points are labeled -1 and drawn with 'x'"""
    X = np.asarray(points, dtype=float)
    labels = np.asarray(labels, dtype=int)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    unique_labels = sorted(set(labels.tolist()))
    for lab in unique_labels:
        mask = labels == lab
        if lab == -1:
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], marker="x")
        else:
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2])

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()



if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)

    ####################
    # 2D (x, y) points
    ######################

    # values to generate clusters that we can visualize
    n_clusters_2d = 8
    pts_per_cluster_2d = 60
    cluster_spread_2d = 0.22  # smaller = tighter clusters
    noise_points_2d = 25

    # create the 2d Cluster points
    centers2d = rng.uniform(low=[-5, -5], high=[15, 10], size=(n_clusters_2d, 2))

    pts2d = []
    for c in centers2d:
        blob = rng.normal(loc=c, scale=cluster_spread_2d, size=(pts_per_cluster_2d, 2))
        pts2d.append(blob)

    pts2d = np.vstack(pts2d)
    noise2d = rng.uniform(low=[-8, -8], high=[18, 13], size=(noise_points_2d, 2))
    pts2d = np.vstack([pts2d, noise2d])

    pts2d_list = [tuple(p) for p in pts2d]

    # run the cluster algorithm
    res2d = cluster_points_2d(pts2d_list, eps=0.6, min_samples=6)
    print("2D clusters:", res2d.n_clusters, "noise:", int(np.sum(res2d.labels == -1)))

    # Plot the clusters
    plot_clusters_2d(pts2d_list, res2d.labels, title="2D DBSCAN")

    ####################
    # 3D (x, y, z) points
    ######################
    n_clusters_3d = 6
    pts_per_cluster_3d = 80
    cluster_spread_3d = 0.28
    noise_points_3d = 30


    # create the clusters and noise that we will classify
    centers3d = rng.uniform(low=[-4, -4, -4], high=[10, 10, 10], size=(n_clusters_3d, 3))

    pts3d = []
    for c in centers3d:
        blob = rng.normal(loc=c, scale=cluster_spread_3d, size=(pts_per_cluster_3d, 3))
        pts3d.append(blob)

    pts3d = np.vstack(pts3d)
    noise3d = rng.uniform(low=[-7, -7, -7], high=[13, 13, 13], size=(noise_points_3d, 3))
    pts3d = np.vstack([pts3d, noise3d])

    pts3d_list = [tuple(p) for p in pts3d]

    # run the clustering function
    res3d = cluster_points_3d(pts3d_list, eps=0.8, min_samples=8)
    print("3D clusters:", res3d.n_clusters, "noise:", int(np.sum(res3d.labels == -1)))

    # visualisze the clusters
    plot_clusters_3d(pts3d_list, res3d.labels, title="3D DBSCAN")
