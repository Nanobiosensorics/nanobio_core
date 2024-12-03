import numpy as np


"""
:param segmentation: microscope image segmentation mask where each cell instance
is marked with an unique number started from 1. (cell ids are 1, 2, 3...)
:return: an array where the ith indexed centroid corresponds to the (i+1)th cell id.
"""
def calculate_microscope_cell_centroids(segmentation: np.ndarray):
  # Calculating the number of cells on the image.
  mx_id = np.max(segmentation)

  # For every cell instance we have a centroid.
  result = []

  # Calculating cell centroids for each cell id.
  for id in range(1, mx_id + 1):
    # Getting x and y indices of the current cell.
    indices = np.where(segmentation == id)

    if len(indices[0]) > 0 and len(indices[1]) > 0:
      # Getting coords by averaging cell indices.
      y_mean = np.mean(indices[0])
      x_mean = np.mean(indices[1])

      result.append(np.array([x_mean, y_mean]))

  return np.array(result)


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

