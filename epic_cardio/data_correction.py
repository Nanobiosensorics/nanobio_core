import numpy as np

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

def corr_data(data):
    corr_data = data.copy()
    for n in range(0, corr_data.shape[0]):
        corr_data[n, :] -= corr_data[n, 1] - corr_data[n, 0]
    for n in range(0, corr_data.shape[0]):
        corr_data[n, :] -= corr_data[n, 0]
    return corr_data

def correct_well(well):
    corr_data = well.copy()
    corr_data -= corr_data[0, :, :]
    well_diff = np.diff(corr_data, axis = 0)
    well_diff_sng = np.max(well_diff, axis = 0)
    corr_data[:, well_diff_sng > np.std(well_diff_sng) * 3] = 0
    corr_data *= 1000
    return corr_data