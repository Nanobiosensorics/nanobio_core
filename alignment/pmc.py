import numpy as np

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
  
  [a_nodes, b_nodes] = np.where(np.linalg.norm(correspondence_vectors[:, np.newaxis] - correspondence_vectors, axis=-1) <= epsilon)

  adjacency_list = [[] for _ in range(len(correspondence_vectors))]
  for a, b in zip(a_nodes, b_nodes): 
    if a != b:
      adjacency_list[a].append(b)
  
  for i in range(len(adjacency_list)):
    adjacency_list[i] = np.array(adjacency_list[i], dtype=np.uint16)

  return correspondence_vectors, adjacency_list


# Gets the candidates which are needed to be evaluated to find the maximal clique.
def __get_required_candidates(S: np.ndarray, F: np.ndarray, gamma: list):
  S_all = np.concatenate((S, F)).astype(np.uint16)      # S U F

  best = S
  for v_i in S_all:
    diff = np.setdiff1d(S, gamma[v_i])                # S \ gamma[v_i]
    if len(diff) < len(best):
      best = diff
  
  return best

# Gets the first available color in a set of colors.
def __get_first_available_color(colors):
  color = 1
  while color in colors:
    color += 1
  return color

# Colors the candidate vectors with greedy coloring algorithm.
def __get_coloring_greedy(S: np.ndarray, gamma: list):
  f = dict()

  for v_i in S:
    used_neighbour_colors = {f[nbr] for nbr in gamma[v_i] if nbr in f}
    f[v_i] = __get_first_available_color(used_neighbour_colors)
  
  return np.array([f[x] for x in S], dtype=np.uint16)

# Updates the coloring to be consecutive numbers if element is removed.
def __update_coloring(f: np.ndarray, idx: int):
  value = f[idx]
  f[idx] = 0

  if value in f:
    return f
  else:
    f[f > value] -= 1
    return f

# def find_translation_pmc(source_points: np.ndarray, target_points: np.ndarray, epsilon: float, correspondence_ratio: float = 1):
#   def __find_clique(candidates: np.ndarray, removed: np.ndarray, coloring: np.ndarray):
#     nonlocal best_clique
#     nonlocal current_clique
#     nonlocal best_clique_len

#     # Selecting the branches which needs to be evaluated.
#     required_candidates = __get_required_candidates(candidates, removed, adjacency_list)

#     i = len(candidates) - 1
#     while i >= 0:
#       # If can't achieve bigger clique then cut this branch.
#       if (len(current_clique) + np.max(coloring)) <= best_clique_len:
#         return

#       node = candidates[i]
#       if node in required_candidates:
#         current_clique = np.append(current_clique, node)

#         new_candidates = np.intersect1d(candidates, adjacency_list[node])
#         if len(new_candidates) > 0:
#           new_removed = np.intersect1d(removed, adjacency_list[node])
#           new_coloring = __get_coloring_greedy(new_candidates, adjacency_list)
#           __find_clique(new_candidates, new_removed, new_coloring)
#         elif len(current_clique) > best_clique_len:
#         #   print(len(current_clique))  
#           best_clique = [current_clique]
#           best_clique_len = len(current_clique)
#         elif len(current_clique) == best_clique_len:
#           best_clique.append(current_clique)
        
        
#         current_clique = current_clique[:-1]
#         candidates = candidates[candidates != node]
#         removed = np.append(removed, node)

#         coloring = __update_coloring(coloring, i)
      
#       i -= 1

#   # Build the graph out of selected indices.
#   correspondence_vectors, adjacency_list = __build_graph(source_points, target_points, epsilon, correspondence_ratio)

#   # Sort the candidates in decreasing degree to minimize branching.
#   candidates = np.argsort([len(x) for x in adjacency_list])[::-1].astype(np.uint16)

#   # Finding the maximal clique.
#   current_clique, best_clique = [], []
#   best_clique_len = 0
#   __find_clique(candidates, np.array([], dtype=np.uint16), __get_coloring_greedy(candidates, adjacency_list))
  
#   translations = [np.average(correspondence_vectors[np.array(clique, dtype=np.uint16)], axis=0) for clique in best_clique]
#   translations = sorted(translations, key=lambda x: np.linalg.norm(x))
# #   translations = [np.average(correspondence_vectors[np.array(best_clique, dtype=np.uint16)], axis=0)]
# #   print(len(best_clique), best_clique_len, translations)
#   return translations[0], len(best_clique)



def find_translation_pmc(source_points: np.ndarray, target_points: np.ndarray, epsilon: float, correspondence_ratio: float = 1):
  def __find_clique(S: np.ndarray, F: np.ndarray, f: np.ndarray):
    nonlocal R_best
    nonlocal R

    # Selecting the branches which needs to be evaluated.
    C = __get_required_candidates(S, F, gamma)

    i = len(S) - 1
    while i >= 0:
      # If can't achieve bigger clique then cut this branch.
      if (len(R) + np.max(f)) <= len(R_best):
        return

      v_i = S[i]
      if v_i in C:
        R = np.append(R, v_i)
        S_new = np.intersect1d(S, gamma[v_i])
        if len(S_new) > 0:
          F_new = np.intersect1d(F, gamma[v_i])
          f_new = __get_coloring_greedy(S_new, gamma)
          __find_clique(S_new, F_new, f_new)
        elif len(R) > len(R_best):
          R_best = R
        
        R = R[:-1]
        S = S[S != v_i]
        F = np.append(F, v_i)

        f = __update_coloring(f, i)
      
      i -= 1

  # Build the graph out of selected indices.
  V, gamma = __build_graph(source_points, target_points, epsilon, correspondence_ratio)

  # Sort the candidates in decreasing degree to minimize branching.
  S = np.argsort([len(x) for x in gamma])[::-1].astype(np.uint16)

  # Finding the maximal clique.
  R, R_best = [], []
  __find_clique(S, np.array([], dtype=np.uint16), __get_coloring_greedy(S, gamma))

  return np.average(V[np.array(R_best, dtype=np.uint16)], axis=0), len(R_best)