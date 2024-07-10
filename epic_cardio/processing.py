from operator import itemgetter
import numpy as np
import os
from nanobio_core.epic_cardio.data_correction import correct_well
from nanobio_core.epic_cardio.cell_selector import WellArrayLineSelector
from nanobio_core.epic_cardio.math_ops import calculate_cell_maximas
from nanobio_core.epic_cardio.defs import WELL_NAMES
from nanobio_core.epic_cardio.measurement_load import load_measurement, wl_map_to_wells, load_high_freq_measurement
from nanobio_core.epic_cardio.defs import *
import json

class RangeType():
    MEASUREMENT_PHASE=0
    INDIVIDUAL_POINT=1

def load_data(path, measurement_type=MeasurementType.TYPE_NORMAL, flip=[False, False]):
    try:
        # Betölti a 3x4-es well képet a projekt mappából.
        wl_map, time = load_measurement(path)
            
        if measurement_type == MeasurementType.TYPE_HIGH_FREQ:
            h_paths = [ (p, os.path.join(path, p), int(p.split('_')[0][6:])) for p in os.listdir(path) if "Cardio" in p]
            h_paths = sorted(h_paths, key=lambda x: x[2])
            
            print(h_paths)
            for name, path, idx in h_paths:
                wl_map_h, time_h = load_high_freq_measurement(path)
                
                wl_map_h -= wl_map_h[0]
                    
                if wl_map is None:
                    wl_map = wl_map_h
                    time = time_h
                else:
                    wl_map_h += np.mean(wl_map[-50:], axis=0)
                    wl_map = np.concatenate([wl_map, wl_map_h])
                    time = np.concatenate([time, time_h[:-1] + 100 + time[-1]])
                
        # Itt szétválasztásra kerülnek a wellek. Betöltéskor egy 240x320-as képen található a 3x4 elrendezésű 12 well.
        wells = wl_map_to_wells(wl_map, flip=flip)
        phases = list(np.where((np.diff(time)) > 60)[0] + 1)
        print([(n+1, p) for n, p in enumerate(phases)])
        return wells, time, phases
    except Exception as e:
        print(f'Error occured during data load: {e}')
        return None

def load_params(path):
    filter_params = {}
    preprocessing, localization = {}, {}
    if os.path.exists(os.path.join(path, '.metadata/parameters.json')):
        with open(os.path.join(path, '.metadata/parameters.json'), 'r') as f:
            obj = json.load(f)
            filter_params = obj['filter_ptss']
            preprocessing = obj['preprocessing']
            localization = obj['localization']
    return filter_params, preprocessing, localization

def save_params(path, well_data, preprocessing, localization):
    if not os.path.exists(os.path.join(path, '.metadata')):
        os.mkdir(os.path.join(path, '.metadata'))
    parameters = {
        # This code iterates through each key-value pair in well_data.
        # And because the json module does not support NumPy integers it converts them to Python integers.
        # For each value[-1], which is a list of tuples, it iterates through each tuple x in the list, 
        # converting each element y within the tuple from a NumPy integer to a Python integer, and reconstructs the tuple with these converted values. 
        'filter_ptss' : {key: [tuple(int(y) for y in x) for x in value[-1]] for key, value in well_data.items()},
        'preprocessing': preprocessing,
        'localization': localization
    }
    with open(os.path.join(path, '.metadata/parameters.json'), 'w+') as f:
        json.dump(parameters, f)


def preprocessing(preprocessing_params, wells, time, phases, background_coords={}):
    well_data = {}
    filter_ptss = {}
    
    if len(preprocessing_params) == 0:
        preprocessing_params = {
            'signal_range' : {
                'range_type': RangeType.MEASUREMENT_PHASE,
                'ranges': [0, None],
            },
            'drift_correction': {
                'threshold': 75,
                'filter_method': 'mean',
                'background_selector': True,
            }
        }
    
    rngs = preprocessing_params['signal_range']['ranges']
    if rngs[1] != None:
        if rngs[0] >= rngs[1]:
            raise ValueError('End point of the range has to be greater than starting point!')

    if preprocessing_params['signal_range']['range_type'] == RangeType.MEASUREMENT_PHASE:
        selected_range = [0 if rngs[0] == 0 or rngs[0] > len(phases) else phases[rngs[0] - 1],
                        len(time) + 1 if rngs[1] == None else phases[rngs[1] - 1]]
    else:   
        selected_range = rngs
        
    slicer = slice(selected_range[0], selected_range[1])
    time = time[slicer]
    phases = [p for p in phases if p >= selected_range[0] and p < selected_range[1]]

    if selected_range[0] > 0:
        tmp = []
        for p in phases:
            if p - selected_range[0] > 0:
                tmp.append(p - selected_range[0])
        phases = tmp

    for name in WELL_NAMES:
        print("Parsing", name, end='\r')
        well_tmp = wells[name]
        
        # if export_params['breakdown_signal']:
        #     line = np.mean(wells[name], axis=(1,2))
        #     peak_until = phases[-1] + np.argmax(line[phases[-1]:])
        #     peak_until = peak_until if line[peak_until] > line[phases[-1] - 1] else phases[-1] - 1
        #     breakdowns[name] = peak_until
        
        well_tmp = well_tmp[slicer]
        well_corr, coords, _ = correct_well(well_tmp, 
                                        coords=[] if len(background_coords) == 0 else background_coords[name],
                                        threshold=preprocessing_params['drift_correction']['threshold'],
                                        mode=preprocessing_params['drift_correction']['filter_method'])
        well_data[name] = well_corr
        filter_ptss[name] = coords
    print("Parsing finished!")
    return well_data, time, phases, filter_ptss, selected_range

def localization(preprocessing_params, localization_params, wells, selected_range, background_coords={}):
    # Sejt szűrés a wellekből.
    well_data = {}
    slicer = slice(selected_range[0], selected_range[1])
    for name in WELL_NAMES:
        print("Parsing", name, end='\r')
        well_tmp = wells[name]
        well_tmp = well_tmp[slicer]
        well_corr, filter_ptss, mask = correct_well(well_tmp, 
                                            coords=[] if not preprocessing_params['drift_correction']['background_selector'] else background_coords[name],
                                            threshold=preprocessing_params['drift_correction']['threshold'],
                                            mode=preprocessing_params['drift_correction']['filter_method'])
        
        ptss = calculate_cell_maximas(well_corr, 
                    min_threshold=localization_params['threshold_range'][0], 
                    max_threshold=localization_params['threshold_range'][1], 
                    neighborhood_size=localization_params['neighbourhood_size'],
                    error_mask=None if not localization_params['error_mask_filtering'] else mask)
        well_data[name] = (well_corr, ptss, filter_ptss)
    print("Parsing finished!")
    return well_data

def parse_selection(well_data:dict, selector:WellArrayLineSelector, evaluation_params:dict) -> (dict, dict):
    selected_ptss = {}
    for name in WELL_NAMES:
        if not evaluation_params['cell_selector']:
            selected_ptss[name] = well_data[name][1]
        else:
            if len(selector.saved_ids[name]) > 0:
                ptss_selected = np.array(itemgetter(*selector.saved_ids[name])(well_data[name][1]), np.uint8)
                if(ptss_selected.ndim == 1):
                    ptss_selected = np.expand_dims(ptss_selected, axis=0)
                selected_ptss[name] = ptss_selected
    return selected_ptss


# fig, axs = plt.subplots(len(lines) // 4, 4, sharex=True, sharey=True)
# fig_2, axs_2 = plt.subplots(len(lines) // 4, 4)

# for n in range(lines.shape[0]):
#     diff = np.diff(np.diff(lines[n]))
#     rng = np.mean(diff) + np.std(diff)
#     err = np.argwhere(np.logical_or(diff > rng, diff < -rng))
#     err = err.reshape((-1,))
#     err.sort()
#     peaks = np.argwhere(np.diff(err) == 1)
#     peaks = peaks.reshape((-1, ))
#     err = np.unique(np.concatenate([err[peaks], err[peaks + 1]]))
#     axs[n // 4, n % 4].plot(lines[n])
#     # for e in err:
#     #     axs[n // 4, n % 4].axvline(e, c='r')    
#     data_range = np.arange(lines[n].shape[0])
#     org = data_range.copy()
#     data_range = np.delete(data_range, err, axis=0)
#     line = np.delete(lines[n], err, axis=0)
#     new_line = np.interp(org, data_range, line)
#     axs_2[n // 4, n % 4].plot(new_line)
# fig.show()
# fig_2.show()