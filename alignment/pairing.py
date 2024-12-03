import numpy as np


"""
    :param source_points: numpy array containing points which will be translated.
    :param target_points: numpy array containing the points where the source points will be translated to.
    :param translation: translation vector to shift source points with.
    :param threshold: pairing maximal difference threshold.
"""
def make_pairing(source_points: np.ndarray, target_points: np.ndarray, translation: np.ndarray, threshold: float):
    # Translate source points.
    translated_source_points = source_points + translation

    # Get all distances between all pairs.
    distances = np.linalg.norm(translated_source_points[:, None, :] - target_points[None, :, :], axis=-1)

    # Get valid pairs by thresholding.
    valid_pairs = np.argwhere(distances <= threshold)
    valid_distances = distances[valid_pairs[:, 0], valid_pairs[:, 1]]

    # Include distance to pairs and sort by it.
    valid_pairs = np.column_stack((valid_pairs, valid_distances))
    valid_pairs = valid_pairs[np.argsort(valid_pairs[:, 2])]

    # Not using the same source and target indices twice to ensure correct pairing.
    used_source_idx, used_target_idx = set(), set()

    # Greedily selecting pairs.
    result = []
    for pair in valid_pairs:
        i, j, diff = int(pair[0]), int(pair[1]), pair[2]
        if i not in used_source_idx and j not in used_target_idx:
            result.append((i, j, diff))
            used_source_idx.add(i)
            used_target_idx.add(j)

    return result

