import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import numpy as np
import matplotlib.pyplot as plt
import cv2
from .data_correction import corr_data

def find_local_maxima(frame, min_threshold=0.1, max_threshold=10, neighborhood_size=5, error_mask=None):
    '''
        Finding local maximum points on a frame.
        
        Parameters
        ----------
        frame - the selected image
    '''
    
    data = frame

    data_max = ndimage.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = ndimage.minimum_filter(data, neighborhood_size)
    diff = np.logical_and(((data_max - data_min) > min_threshold),((data_max - data_min) < max_threshold))
    maxima[diff == 0] = 0
    
    if error_mask is not None:
        error_distance = ndimage.distance_transform_edt(np.logical_not(error_mask))
        maxima[np.logical_and(error_distance <= 2, maxima == 1)] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
    return x, y

def display_local_maxima(frame, coords, col='r', annotate=True, both=False):
    '''
        Draw the maximum points onto an image.
        
        Parameters
        ----------
        frame  - the selected image
        coords - the coordicates of the maximum points
        col    - the color palette of the points
    '''
    # import matplotlib.pyplot as plt
    if both:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 10))
        ax1.imshow(frame)
        ax2.imshow(frame)
        ax2.autoscale(False)
        ax2.scatter(coords[0],coords[1], c=col)
        if annotate:
            annotation = [str(n) for n in range(len(coords[0]))]
            for i, txt in enumerate(annotation):
                ax2.annotate(txt, (coords[0][i] + 1, coords[1][i]))
        plt.show()
    else:
        plt.figure(figsize=(20, 10))
        plt.imshow(frame)
        plt.autoscale(False)
        plt.scatter(coords[0],coords[1], c=col)
        if annotate:
            annotation = [str(n) for n in range(len(coords[0]))]
            for i, txt in enumerate(annotation):
                plt.annotate(txt, (coords[0][i] + 1, coords[1][i]))
        plt.show()
        
def display_local_maxima_highlight_selected(frame, coords, selected, col=['r', 'g'], annotate=True):
    '''
        Draw the maximum points onto an image and highlight selected ones.
        
        Parameters
        ----------
        frame    - the selected image
        coords   - the coordicates of the maximum points
        selected - index array which contains the indices of the selected points
        col      - the color palette of the points
    '''
    # import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 10))
    plt.imshow(frame)
    plt.autoscale(False)
    for i in range(len(coords[0])):
        if i in selected:
            plt.scatter(coords[0][i],coords[1][i], c=col[1])
        else:
            plt.scatter(coords[0][i],coords[1][i], c=col[0])
    if annotate:
        annotation = [str(n) for n in range(len(coords[0]))]
        for i, txt in enumerate(annotation):
            plt.annotate(txt, (coords[0][i] + 1, coords[1][i]))
    plt.show()

def get_coords_from_local_maxima(well):
    # Minden frame-re kiszámoltatom a maximum helyeket.
    xs, ys = [], []
    for i in range(well.shape[0]):
        x,y = find_local_maxima(well[i, :, :], min_threshold=.08, max_threshold=.4)
        xs.append(x)
        ys.append(y)
    # A koordinátákat berendeze egy (x, y, idő) oszlopokból álló táblába.
    df = []
    t = np.linspace(0, well.shape[0], well.shape[0])
    for x, y, ct in zip(xs, ys, t):
        for dx, dy in zip(x,y):
            df.append([dx, dy, ct])
    return np.array(df)

def get_dilated_mask(coords, kernel_size=(3,3)):
    # Maszk készítése a kiválasztott pontok alapján.
    mask = np.ones((80,80))
    pt_mask = np.zeros((80, 80))
    for pt in coords:
        pt_mask[int(pt[1]), int(pt[0])] = 1
    # A pontok környezetének maszkolása, hogy a sejt környezete
    # ne kerüljön a háttérszámításba.
    kernel = np.ones(kernel_size, np.uint8)
    dil = cv2.dilate(pt_mask, kernel, iterations=1)
    dil = np.logical_not(dil)
    dil = np.logical_and(mask, np.logical_not(dil))
    return dil

def get_avg_signal(well, mask=None):
    if mask is None:
        mask = np.ones(well.shape[1:])
    mask = np.logical_not(mask)
    mask_array = np.repeat(mask[np.newaxis, :, :], well.shape[0], axis=0)
    well_masked = np.ma.array(well, mask = mask_array)
    return np.mean(well_masked, axis=(1,2))

def max_center_signals(signals, window_size=100):
    tr_signals = signals.copy()
    tr_signals = corr_data(tr_signals)
    max_points = list(np.argmax(tr_signals, axis=1))
    max_points = np.array([ (idx, point) for idx, point in enumerate(max_points) if tr_signals.shape[1] - point > window_size ])
    if max_points.shape[0] == 0:
        return None
    mx = np.max(tr_signals.shape[1] - max_points[:, 1])
    max_mtx = np.zeros((max_points.shape[0], mx*2 + 1))
    for i, (idx, pt) in enumerate(max_points):
        start  = int(np.round(max_mtx.shape[1] / 2) - pt) if np.round(max_mtx.shape[1] / 2) - pt > 0 else 0
        end = int(np.round(max_mtx.shape[1] / 2) - pt + tr_signals.shape[1]) if np.round(max_mtx.shape[1] / 2) - pt + tr_signals.shape[1] < max_mtx.shape[1] else max_mtx.shape[1]
        max_mtx[i, start:end] = tr_signals[idx, tr_signals.shape[1] - (end - start):tr_signals.shape[1]]
    max_mtx = corr_data(max_mtx)
    if max_mtx.ndim == 1:
        max_mtx = np.expand_dims(max_mtx, axis=0)
    return max_mtx