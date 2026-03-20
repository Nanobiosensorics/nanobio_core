import numpy as np


"""
Compute centroids for labeled cell instances in a 2D segmentation mask.

:param segmentation: 2D numpy array of shape (H, W) containing integer labels,
    where 0 represents background and each cell instance is assigned a unique
    positive integer ID (1, 2, 3, ...).

:return: numpy array of shape (N, 2) containing centroid coordinates for each
    labeled cell, where N is the number of unique cell IDs (excluding background).
    Each row corresponds to a cell ID in ascending order, such that the ith row
    contains the centroid of cell with label (i + 1). Coordinates are returned
    in (x, y) format as floating-point values.
"""
def calculate_microscope_cell_centroids(segmentation: np.ndarray):
    labels = segmentation.ravel()

    # Get coordinates
    y_coords, x_coords = np.indices(segmentation.shape)
    y_coords = y_coords.ravel()
    x_coords = x_coords.ravel()

    # Count pixels per label
    counts = np.bincount(labels)

    # Sum coordinates per label
    sum_x = np.bincount(labels, weights=x_coords)
    sum_y = np.bincount(labels, weights=y_coords)

    # Avoid division by zero
    valid = counts > 0

    # Compute centroids
    centroids_x = sum_x[valid] / counts[valid]
    centroids_y = sum_y[valid] / counts[valid]

    # Stack results (skip background label 0)
    return np.stack((centroids_x[1:], centroids_y[1:]), axis=1)


"""
:param segmentation: microscope image segmentation mask where each cell instance
is marked with an unique number started from 1. (cell ids are 1, 2, 3...)
:return: an identical mask where each centroid pixel is marked with 1.
"""
def get_microscope_cell_centroids_mask(segmentation: np.ndarray):
  cell_centroids = calculate_microscope_cell_centroids(segmentation)
  mask = np.zeros_like(segmentation)
  for centroid in cell_centroids:
    x, y = int(centroid[0]), int(centroid[1])
    mask[y][x] = 1
  return mask

