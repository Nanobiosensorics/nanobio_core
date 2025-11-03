import cProfile
from io import StringIO
from typing import List, Set

import numpy as np
from scipy.spatial import cKDTree

"""
:param source_points: numpy array containing points which will be translated. (-1, 2) shaped.
:param target_points: numpy array containing the points where the source points will be translated to. (-1, 2) shaped.
:param epsilon: pairwise consistency threshold.
:param correspondence_ratio: the percentage of random correspondence vector samples.
"""

# Builds a graph out of correspodence vectors between source and target points with epsilon maximal threshold.
def __build_graph(source_points: np.ndarray, target_points: np.ndarray, epsilon: float, correspondence_ratio: float):
  # Random sampling.
  correspondence_vectors = (target_points - source_points[:, np.newaxis]).reshape(-1, 2)
  correspondence_vectors = correspondence_vectors[np.random.choice(len(correspondence_vectors), int(correspondence_ratio * len(correspondence_vectors)), replace=False)]

  # Find vector pairs within epsilon without materializing the full pairwise distance matrix.
  tree = cKDTree(correspondence_vectors)
  neighbors = tree.query_ball_tree(tree, r=epsilon)

  a_list = []
  b_list = []
  for i, nbrs in enumerate(neighbors):
    if not nbrs:
      continue
    a_list.extend([i] * len(nbrs))
    b_list.extend(nbrs)

  if a_list:
    a_nodes = np.array(a_list, dtype=np.uint32)
    b_nodes = np.array(b_list, dtype=np.uint32)
  else:
    a_nodes = np.array([], dtype=np.uint32)
    b_nodes = np.array([], dtype=np.uint32)

  adjacency_list: List[Set[int]] = [set() for _ in range(len(correspondence_vectors))]
  for a, b in zip(a_nodes, b_nodes): 
    if a != b:
      adjacency_list[a].add(int(b))

  return correspondence_vectors, adjacency_list


# Gets the candidates which are needed to be evaluated to find the maximal clique.
def __get_required_candidates(S: List[int], F: Set[int], gamma: List[Set[int]]):
  S_set = set(S)
  best = S
  min_size = len(best)
  for v_i in S_set.union(F):
    diff = S_set.difference(gamma[v_i])
    size = len(diff)
    if size < min_size:
      if size == 0:
        return set()
      min_size = size
      diff_set = diff
      best = [node for node in S if node in diff_set]
  return set(best)

# Gets the first available color in a set of colors.
def __get_first_available_color(colors):
  color = 1
  while color in colors:
    color += 1
  return color

# Colors the candidate vectors with greedy coloring algorithm.
def __get_coloring_greedy(S: List[int], gamma: List[Set[int]]):
  f = {}
  coloring = []
  for v_i in S:
    used_neighbour_colors = {f[nbr] for nbr in gamma[v_i] if nbr in f}
    color = __get_first_available_color(used_neighbour_colors)
    f[v_i] = color
    coloring.append(color)
  return coloring

# Updates the coloring to be consecutive numbers if element is removed.
def __update_coloring(f: List[int], idx: int):
  value = f[idx]
  f.pop(idx)
  if value in f:
    return f
  for i, color in enumerate(f):
    if color > value:
      f[i] = color - 1
  return f

def find_translation_pmc(source_points: np.ndarray, target_points: np.ndarray, epsilon: float, correspondence_ratio: float = 1):
  def __find_clique(S: List[int], F: Set[int], f: List[int]):
    nonlocal R_best
    nonlocal R

    # Selecting the branches which needs to be evaluated.
    C = __get_required_candidates(S, F, gamma)

    i = len(S) - 1
    while i >= 0:
      # If can't achieve bigger clique then cut this branch.
      max_color = max(f) if f else 0
      if (len(R) + max_color) <= len(R_best):
        return

      v_i = S[i]
      if v_i in C:
        R.append(v_i)
        adjacency = gamma[v_i]
        S_new = [node for node in S if node in adjacency]
        if len(S_new) > 0:
          f_new = [color for node, color in zip(S, f) if node in adjacency]
          F_new = F.intersection(adjacency)
          __find_clique(S_new, F_new, f_new)
        elif len(R) > len(R_best):
          R_best = list(R)

        R.pop()
        S.pop(i)
        F = F.union({v_i})

        f = __update_coloring(f, i)
      
      i -= 1

  # Build the graph out of selected indices.
  V, gamma = __build_graph(source_points, target_points, epsilon, correspondence_ratio)

  # Sort the candidates in decreasing degree to minimize branching.
  S = sorted(range(len(gamma)), key=lambda idx: len(gamma[idx]), reverse=True)

  # Finding the maximal clique.
  R, R_best = [], []
  __find_clique(S, set(), __get_coloring_greedy(S, gamma))

  return np.average(V[np.array(R_best, dtype=np.uint16)], axis=0), len(R_best)