import numpy as np
from .cell_maxima import get_coords_from_local_maxima, get_dilated_mask, get_avg_signal, find_local_maxima
from scipy.signal import argrelextrema
from skimage.restoration import denoise_tv_chambolle
from scipy.signal import find_peaks
import cv2

def get_unique_coords(data):
    '''
       Selects the unique coordinates from the 3 dimensional coordinate matrix.
       
       Parameters
       ----------
       data - the (t, x, y) dimensional coord matrix
    '''
    def contains_coord(array, x, y):
        for ax, ay in array:
            if ax == x and ay == y:
                return True
        return False
    
    coords = []
    for x, y in data:
        if not contains_coord(coords, x, y):
            coords.append((x, y))
    return coords

def get_coord_frequency(data, unique):
    '''
        Calculates the frequency of the point selection.
        
        Parameters
        ----------
        data   - the (t, x, y) dimensional coord matrix
        unique - array of the unique points
    '''
    freq = []
    for x, y in unique:
        i = 0
        for dx, dy in data:
            if x == dx and y == dy:
                i+=1
        freq.append((x, y, i))
    return freq

def euclid_dist(p1, p2):
    '''
        Calculates the Euclidean distance between two points.
        
        Parameters
        ----------
        p1 - coordinates of the first point
        p2 - coordinates of the second point
    '''
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def get_adjacency_matrix(coords):
    '''
        Calculates the adjacency matrix of the maximum points.
        
        Parameters
        ----------
        coords - the unique maximum points
    '''
    dist = np.zeros((len(coords),len(coords)))
    for i, p1 in enumerate(coords):
        for j, p2 in enumerate(coords):
            dist[i, j] = 1 if euclid_dist(p1, p2) == 1 else 0
    return dist

def get_cluster_centeroid(arr):
    '''
        Calculates the centroid of a point cluster.
        
        Parameters
        ----------
        arr - list of points
    '''
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def get_adj_cluster(p, unique_coords, adj_mtx):
    '''
        Get the point clusters of the neighbouring points.
        
        Parameters
        ----------
        p - the point array
        unique_coords - unique coords in the point array
        adj_mtx - adjacency matrix
    '''
    def get_neighbours(dist, i):
        n = []
        for j in range(dist.shape[1]):
            if dist[i, j]:
                yield unique_coords[j]
    group = []
    group.append(p)
    while True:
        update = 0
        temp = []
        for p in group:
            for pn in get_neighbours(adj_mtx, unique_coords.index(p)):
                if pn not in group:
                    temp.append(pn)
        if len(temp) == 0:
            break
        else:
            for pn in set(temp):
                group.append(pn)
    return group

def get_most_selected(coords, freq):
    selected_coords_freq = []
    for coord_freq in freq:
        if (coord_freq[0], coord_freq[1]) in coords:
            selected_coords_freq.append(coord_freq)
    return selected_coords_freq[np.argmax(np.asarray(selected_coords_freq)[:, 2])][:2]


def calculate_cells(well):
    coords = get_coords_from_local_maxima(well)
    xy = coords[:, :2]
    uxy = get_unique_coords(xy)
    mask = get_dilated_mask(uxy)
    freq = get_coord_frequency(xy, uxy)
    dist = get_adjacency_matrix(uxy)
    # dist_df = pd.DataFrame(dist, columns=uxy, index=uxy)

    well_corr = well.copy()
    # Az átlag háttérérték kiszámítása minden képkockán
    avgs = get_avg_signal(well, mask)
    # A háttér levonása az eredeti wellből.
    for i in range(well_corr.shape[0]):
        well_corr[i, :, :] = well[i, :, :] - avgs[i]


    # Az átlagvonal
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(avgs)
    # plt.show()
    # 3 dimenzióban kirajzolom, az aktív pontok térbeli és időbeli elhelyezkedését
    # data = pd.DataFrame(coords, columns=['X', 'Y', 'T'])
    # display_active_coords(data)

    # A cell start paraméter az alapján állapítható meg, hogy 
    # milyen időpontban jelennek meg sejtek a 3D-s grafikonon
    CELL_START = 0
    # Releváns régió levágása
    well_corr = well_corr[CELL_START:, :, :]

    adj = []
    points = uxy.copy()
    selected = []
    # Az egyedi pontokon végigmegy és lekérdezi 
    # a pontokhoz tartozó szomszédossági klasztert.
    # Minden iterációban a iteráció pontja és a klaszterébe
    # tartozó összes pont kiválasztásra kerül és ezt követően
    # a kiválasztott pontok szomszédjai már nem kerülnek vizsgálatra
    # mivel a klasztereik megegyeznének az iterált pontéval.
    for p in points:
        if p not in selected:
            temp = get_adj_cluster(p, uxy, dist)
        else:
            continue
        for p2 in temp:
            if p2 not in selected:
                selected.append(p2)
        adj.append(temp)
        
    # Minden klaszterre kiszámolja a centroidot.
    centroids = []
    max_coords = []
    for arr in adj:
        centroids.append(get_cluster_centeroid(np.asarray(arr)))
        max_coords.append(get_most_selected(arr, freq))
    max_arr = np.asarray(max_coords)
    cent_arr = np.asarray(centroids)

    # Az egy klaszterbe tartozó pontok frekvenciájinak összegzése.
    adj_freq = []
    for point, cent in zip(adj, cent_arr):
        total = 0
        for p in point:
            idx = uxy.index(p)
            total += freq[idx][2]
        adj_freq.append((total, cent))

    # A klaszterek redukciója a frekvencia alapján.
    adj_300 = []
    for fr, cent in adj_freq:
        if fr <= 300:
            adj_300.append(cent)
    pts = np.round(np.asarray(adj_300))

    # A kiválasztott pontokhoz tartozó pixelértékek kiválasztása.
    lines = []
    for pt in pts:
        lines.append(well_corr[:, int(pt[1]), int(pt[0])])
    lines = np.asarray(lines).T

    ptss = [ pts[i, :] for i in range(lines.shape[1]) if np.mean(np.diff(lines[:, i])) > 0 and not np.any(lines[:, i] < -.5) ]
    lines = [ lines[:, i] for i in range(lines.shape[1]) if np.mean(np.diff(lines[:, i])) > 0 and not np.any(lines[:, i] < -.5) ]

    # plt.figure(figsize=(20, 10))
    # for i in range(lines.shape[1]):
    #         plt.plot(lines[:, i])
    # plt.show()

    return well_corr, ptss, lines


def calculate_cell_maximas(data, **kwargs):
    data_mx = np.max(data, axis = 0)
    return np.array(find_local_maxima(data_mx, **kwargs), dtype=np.int32).T

def standardize_signal(signal):
    return (signal - signal.mean()) / signal.std()
    
def destandardize_signal(signal, mean, std):
    return (signal * std) + mean

def get_max_well(well):
    return np.max(well, axis = 0)

def signal_segments(signal):
    x = np.array(signal).astype('float')
    x_std = standardize_signal(x)
    x_denoise = denoise_tv_chambolle(x_std, weight=.05)  # adjust the parameters
    x_denoise = destandardize_signal(x_denoise, x.mean(), x.std())
    # grads = np.gradient(x_denoise)

    # def exp(x, arr):
    #     return x if x > np.std(arr) / 5 or x < -np.std(arr) / 5 else 0
    # def array_for(x):
    #     return np.array([exp(xi, x) for xi in x])
    # grads_filter = grads.copy()
    # grads_filter = array_for(grads_filter)

    # max_ind = argrelextrema(grads_filter, np.greater)
    # min_ind = argrelextrema(grads_filter, np.less)
    # ids = np.hstack((max_ind, min_ind)).squeeze()

    # l = []
    # for n in ids:
    #     env = list(range(n-2, n+2+1))
    #     env.remove(n)
    #     env = [i for i in env if i >= 0 and i < x.shape[0]]
    #     if np.abs(np.mean(x[env]) - x[n]) > 0.1:
    #         l.append(n)
    # ids = l
    # coords = np.vstack((ids, grads[ids])).T
    peaks = find_peaks(np.abs(np.diff(x_denoise)), height = np.std(np.diff(x_denoise)) * 2)[0]
    return peaks

def flatten(t):
    return [item for sublist in t for item in sublist]

def reduce_adjacent_elements(elements):
    red = elements.copy()
    red.sort()
    red = np.array(red)
    red = np.hstack([red[0], red[np.add(np.where(np.logical_not(np.diff(red) < 5))[0], 1)]])
    return list(red)

def signal_jumps_in_well(well):
    mn = np.mean(well, axis=(1,2))
    return list(signal_segments(mn))

def total_signal_jumps_in_measurement(measurement):
    WELL_NAMES = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3', 'C4']
    jmps = []
    for name in WELL_NAMES:
        mn = np.mean(measurement[name], axis=(1,2))
        coords = list(signal_segments(mn))
        jmps.append(coords)
    jmps = list(set(flatten(jmps)))
    jmps = reduce_adjacent_elements(jmps)
    return jmps

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized