import numpy as np
import os
import pandas as pd
import re
from string import ascii_uppercase
    
def load_measurement(dir_path):
    '''
        Load the well image from a project folder created by the Epic Cardio. 
        Required folders/files are the DMR folder, the test_WL_Power and the
        test_avg file.
        
        Parameters
        ----------
        dir_path: the path to the folder
    '''
    S = 0.0002
    files = [ obj for obj in os.listdir(dir_path) if '_wl_power' in obj.lower()]
    
    if len(files) == 0:
        print("Missing test wl power file!!!")
        return None, None
    
    wl_power_path = os.path.join(dir_path, files[0])
    
    files = [ obj for obj in os.listdir(dir_path) if '_avg' in obj.lower()]
    
    if len(files) == 0:
        print("Missing test avg!!!")
        return None, None
    
    avg_path = os.path.join(dir_path, files[0])
    
    meas_mode = 'normal' if os.path.getsize(wl_power_path) == 614400 else 'high'
    
    print(f"Measurement mode {meas_mode}")
    
    fr = open( wl_power_path, "rb")
    init_map = np.frombuffer(fr.read(614400), dtype='float32' if meas_mode == 'normal' else 'uint16')
    print(init_map.shape)
    init_wl_map = np.reshape(init_map[:76800], [240, 320])
    fr.close()

    sorted_files = os.listdir(os.path.join(dir_path, 'DMR'))
    
    if len(sorted_files) <= 0:
        print('DMR folder is empty!');
        return None 
    
    sorted_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    timestep_mats = np.zeros([len(sorted_files),240,320])
    for i in range(len(sorted_files)):
        step = open(os.path.join(dir_path, f'DMR/{i + 1}'), 'rb')
        A_int = np.frombuffer(step.read(153600), dtype='uint16' if meas_mode == 'normal' else 'uint8')
        timestep_mats[i,:,:] = np.reshape(A_int,[240,320])
        step.close()

    WL_map = np.tile(init_wl_map, [len(timestep_mats),1, 1]) + S*(timestep_mats-np.tile(timestep_mats[0,:,:],[len(timestep_mats),1,1]))

    time = []
    try:
        time = pd.read_table(avg_path, skiprows=1, decimal=',')
        time = np.asarray(time.iloc[:,0]) * 60
    except Exception as e:
        time = np.linspace(0, WL_map.shape[0] * 9, WL_map.shape[0])
    print(f"Measurement loaded {WL_map.shape} time {time.shape}")
    return WL_map, time

def load_high_freq_measurement(dir_path):
    '''
        Load the well image from a project folder created by the Epic Cardio. 
        Required folders/files are the DMR folder, the test_WL_Power and the
        test_avg file.
        
        Parameters
        ----------
        dir_path: the path to the folder
    '''
    S = 0.002
    files = [ obj for obj in os.listdir(dir_path) if '_ms' == obj.lower()[-3:]]
    
    if len(files) == 0:
        print("Missing test wl power file!!!")
        return None
    
    wl_power_path = os.path.join(dir_path, files[0])
    
    files = [ obj for obj in os.listdir(dir_path) if '_avg' in obj.lower()]
    
    if len(files) == 0:
        print("Missing test avg!!!")
        return None
    
    avg_path = os.path.join(dir_path, files[0])
    
    # print(wl_power_path)
    # print(avg_path)
    
    fr = open( wl_power_path, "rb")
    init_map = np.frombuffer(fr.read(614400), dtype='uint16')
    # print(init_map.shape)
    init_wl_map = np.reshape(init_map[:76800], [240, 320])
    fr.close()

    sorted_files = os.listdir(os.path.join(dir_path, 'fDMR'))
    
    if len(sorted_files) <= 0:
        print('DMR folder is empty!');
        return None 
    
    sorted_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    timestep_mats = np.zeros([len(sorted_files),240,320])
    for i in range(len(sorted_files)):
        step = open(os.path.join(dir_path, f'fDMR/{i + 1}'), 'rb')
        A_int = np.frombuffer(step.read(153600), dtype='uint8')
        timestep_mats[i,:,:] = np.reshape(A_int,[240,320])
        step.close()

    WL_map = np.tile(init_wl_map, [len(timestep_mats),1, 1]) + S*(timestep_mats-np.tile(timestep_mats[0,:,:],[len(timestep_mats),1,1]))

    time = []
    time = pd.read_table(avg_path, skiprows=1, decimal=',')
    time = np.asarray(time.iloc[:,0])
    # print(f"Measurement loaded {WL_map.shape} time {time.shape}")
    return WL_map, time

def load_measurement_bt(dir_path):
    S = 0.0002
    filename = dir_path + '/240x360x4x3_test_WL_Power'

    filename = filename if os.path.exists(filename) else dir_path + '/240x360x5x3_test_WL_Power'

    fr = open( filename, "rb")
    init_map = np.frombuffer(fr.read(691200), dtype='float32')
    init_wl_map = np.reshape(init_map[:86400], [240, 360])
    fr.close()

    sorted_files = os.listdir(dir_path + '/DMR')
    
    if len(sorted_files) <= 0:
        print('DMR folder is empty!');
        return None 
    
    sorted_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    timestep_mats = np.zeros([len(sorted_files),240,360])
    for i in range(len(sorted_files)):
        step = open(dir_path + f'/DMR/{i + 1}', 'rb')
        A_int = np.frombuffer(step.read(172800), dtype='uint16')
        step.close()
        timestep_mats[i,:,:] = np.reshape(A_int,[240,360])

    WL_map = np.tile(init_wl_map, [len(timestep_mats),1, 1]) + S*(timestep_mats-np.tile(timestep_mats[0,:,:],[len(timestep_mats),1,1]))
    
    time = []
    if os.path.exists(dir_path + '/test_avg'):
        time = pd.read_table(dir_path + '/test_avg', skiprows=1, decimal=',')
        time = np.asarray(time.iloc[:,0]) * 60
    return WL_map, time

def load_measurement_384w(dir_path):
    '''
        Load the well image from a project folder created by the Epic Cardio. 
        Required folders/files are the DMR folder, the test_WL_Power and the
        test_avg file.
        
        Parameters
        ----------
        dir_path: the path to the folder
    '''
    S = 0.0002
    files = [ obj for obj in os.listdir(dir_path) if 'wl_power' in obj.lower()]
    
    if len(files) == 0:
        print("Missing test wl power file!!!")
        return None
    
    wl_power_path = os.path.join(dir_path, files[0])
    
    files = [ obj for obj in os.listdir(dir_path) if '_avg' in obj.lower()]
    
    avg_path = None
    if len(files) > 0:
        os.path.join(dir_path, files[0])
    else:
        print("Missing test avg!!!")
    
    fr = open( wl_power_path, "rb")
    init_map = np.frombuffer(fr.read(), dtype=np.float32)
    init_wl_map = np.reshape(init_map[:345600], [480, 720])
    fr.close()

    sorted_files = os.listdir(os.path.join(dir_path, 'DMR'))
    
    if len(sorted_files) <= 0:
        print('DMR folder is empty!');
        return None 
    
    sorted_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    timestep_mats = np.zeros([len(sorted_files),480,720])
    for i in range(len(sorted_files)):
        step = open(os.path.join(dir_path, f'DMR/{i + 1}'), 'rb')
        A_int = np.frombuffer(step.read(), dtype=np.uint16)
        timestep_mats[i,:,:] = np.reshape(A_int,[480,720])
        step.close()

    WL_map = np.tile(init_wl_map, [len(timestep_mats),1, 1]) + S*(timestep_mats-np.tile(timestep_mats[0,:,:],[len(timestep_mats),1,1]))

    if avg_path != None:
        time = []
        time = pd.read_table(avg_path, skiprows=1, decimal=',')
        time = np.asarray(time.iloc[:,0]) * 60
    else:
        time = np.array(list(range(0, WL_map.shape[0])))
    print(f"Measurement loaded {WL_map.shape} time {time.shape}")
    return WL_map, time

def wl_map_to_wells(wl_map, flip=[False, False]):
    WIDTH = 80
    WELL_IDS = [['C1', 'C2', 'C3', 'C4'], ['B1', 'B2', 'B3', 'B4'], ['A1', 'A2', 'A3', 'A4']]
    flip_axis = list(np.where(np.array(flip) == True)[0] + 1)
    return {name: np.flip(wl_map[:, i : i+WIDTH, j:j+WIDTH], axis=flip_axis) if len(flip_axis) > 0 else wl_map[:, i : i+WIDTH, j:j+WIDTH] 
            for names, i in zip(WELL_IDS, range(0, 240, WIDTH)) for name, j in zip(names, range(0, 320, WIDTH))}

def wl_map_to_wells_384w(wl_map, flip=False, padding = 4):
    from string import ascii_uppercase
    WIDTH = 30
    grid = []
    for i in range(16):
        row = []
        for j in range(24):
            row.append(ascii_uppercase[i] + str(j))
        grid.append(row)
    return {name: np.flip(wl_map[:, i + padding : i+WIDTH - padding - 1, j + padding:j + WIDTH - padding], axis=1) if flip else wl_map[:, i + padding : i+WIDTH - padding - 1, j + padding:j+WIDTH - padding] 
            for names, i in zip(grid, range(0, 480, WIDTH)) for name, j in zip(names, range(0, 720, WIDTH))}