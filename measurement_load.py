import numpy as np
import os
import pandas as pd
import re
    
    
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
    filename = dir_path + '/240x320x4x3_test_WL_Power'

    filename = filename if os.path.exists(filename) else dir_path + '/240x320x5x3_test_WL_Power'

    fr = open( filename, "rb")
    init_map = np.frombuffer(fr.read(614400), dtype='float32')
    init_wl_map = np.reshape(init_map[:76800], [240, 320])
    fr.close()

    sorted_files = os.listdir(dir_path + '/DMR')
    sorted_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    timestep_mats = np.zeros([len(sorted_files),240,320])
    for i in range(len(sorted_files)):
        step = open(dir_path + f'/DMR/{i + 1}', 'rb')
        A_int = np.frombuffer(step.read(153600), dtype='uint16')
        step.close()
        timestep_mats[i,:,:] = np.reshape(A_int,[240,320])

    WL_map = np.tile(init_wl_map, [len(timestep_mats),1, 1]) + S*(timestep_mats-np.tile(timestep_mats[0,:,:],[len(timestep_mats),1,1]))

    time = []
    if os.path.exists(dir_path + '/test_avg'):
        time = pd.read_table(dir_path + '/test_avg', skiprows=1, decimal=',')
        time = np.asarray(time.iloc[:,0]) * 60
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