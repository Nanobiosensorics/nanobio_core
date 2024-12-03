import numpy as np


"""
Calculates the optimal affine transformation with least squares.
:param source_points: numpy array containing points which will be translated.
:param target_points: numpy array containing the points where the source points will be translated to.
:param pairing: the pairs to minimize distance between.
"""
def get_affine_transformation(source_points: np.ndarray, target_points: np.ndarray, pairing: list):
    n = len(pairing)

    # Constructing homogeneous source point and target points from pairing.
    P_hom, Q = np.zeros((n, 3)), np.zeros((n, 2))
    for i in range(n):
        P_hom[i] = np.append(source_points[pairing[i][0]], 1)
        Q[i] = target_points[pairing[i][1]]
    
    # Constructing and solving the linear equation system.
    A, b = np.zeros((2 * n, 6)), np.zeros(2 * n)
    for i in range(n):
        A[2 * i, :3] = P_hom[i]
        b[2 * i] = Q[i, 0]
        A[2 * i + 1, 3:] = P_hom[i]
        b[2 * i + 1] = Q[i, 1]
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Converting params to matrix and transpone to work with row vector.
    affine = np.append(params, [0, 0, 1]).reshape(3, 3)
    return affine.T

