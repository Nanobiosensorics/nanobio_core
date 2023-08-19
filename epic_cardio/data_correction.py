import numpy as np
from scipy.ndimage import distance_transform_edt

# def correct_data(data,idx,length):
#     '''
#         Clear offsets off of a signal so it starts in 0.
        
#         Parameters
#         ----------
#         data   - a table which contains the signals
#         idx    - that start index of the signal
#         length - the length of the output signals
#     '''
#     def shift_jump(data,idx):
#         data_shifted = data.copy()
#         for cell in data.columns:
#             data_shifted[cell][idx:]=data_shifted[cell][idx:]-(data_shifted[cell][idx]-data_shifted[cell][idx-1])
#         return data_shifted
    
#     def shift_to_zero(data,idx):
#         data_shifted = data.copy()
#         for cell in data.columns:
#             data_shifted[cell]=data_shifted[cell]-data_shifted[cell][idx]
#         return data_shifted
    
#     data_corr = shift_jump(data,idx)
#     data_corr = shift_to_zero(data_corr,idx)
#     data_corr = data_corr.iloc[idx:idx+length,:].reset_index(drop=True)
#     data_corr = np.abs(data_corr)
#     return data_corr

def select_indices(bool_array, num_indices, spacing = 2, random_distance=5):
    # Get the indices of the true elements in the boolean array
    true_indices = np.argwhere(bool_array)
    dst = distance_transform_edt(bool_array)
    
    rows, cols = bool_array.shape
    center_row = rows // 2
    center_col = cols // 2
    
    distances_to_center = np.array([np.sqrt((y - center_row)**2 + (x - center_col)**2) for y, x in true_indices])
    preference_weights = 1 / (distances_to_center + 1)  # Adding 1 to avoid division by zero
    
    chosen_indices = []
    
    while len(chosen_indices) < num_indices:
        # Choose a random true index
        random_true_index = np.random.choice(len(true_indices), p=preference_weights.ravel() / np.sum(preference_weights))
        random_true_index = np.unravel_index(random_true_index, (rows, cols))
        
        # Check if the chosen index is at least 'random_distance' units away from other chosen indices
        if all(np.linalg.norm(np.array(chosen_index) - random_true_index) >= random_distance for chosen_index in chosen_indices) and dst[*random_true_index] > spacing:
            chosen_indices.append(random_true_index)
    
    # Split the chosen indices into x and y groups
    x_indices, y_indices = zip(*chosen_indices)
    
    return x_indices, y_indices

def corr_data(data):
    corr_data = data.copy()
    for n in range(0, corr_data.shape[0]):
        corr_data[n, :] -= corr_data[n, 1] - corr_data[n, 0]
    for n in range(0, corr_data.shape[0]):
        corr_data[n, :] -= corr_data[n, 0]
    return corr_data

def correct_well(well, threshold = 50):
    corr_data = well.copy()
    corr_data -= corr_data[0, :, :]
    
    well_diff = np.diff(corr_data, axis = 0)
    well_diff_sng = np.max(well_diff, axis = 0)
    corr_data[:, well_diff_sng > np.std(well_diff_sng) * 3] = 0
    corr_data *= 1000
    
    mask = corr_data[-1] < threshold
    
    indices = select_indices(mask, 10, 2, 3)
    
    fltr = np.transpose(np.tile(np.mean(corr_data[:, indices[1], indices[0]], axis=1), (80, 80, 1)), (2,0,1))
    
    corr_data -= fltr
    corr_data -= corr_data[0, :, :]
    corr_data[corr_data < 0] = 0
    
    return corr_data