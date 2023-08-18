import numpy as np
import matplotlib
from matplotlib.patches import Polygon
import cv2

def centroid(x,y):
    length = x.shape[0]
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    return sum_x/length, sum_y/length

def mask_centers(mask, markers=None):
    coords = []
    it = np.unique(mask)[1:] if markers == None else markers
    for i in it:
        y,x = (mask == i).nonzero()
        coords.append(centroid(x,y))
    coords = np.asarray(coords)
    return coords

def euclid(a, b):
    return np.linalg.norm(a-b)

def get_coord_from_idx(idx):
    x = np.floor(idx / 80)
    y = idx - x * 80
    return int(x), int(y)

def get_closest_coords(cd, markers):
    closest_idx = 0
    dist = euclid(markers[closest_idx, :], cd)
    closest_coords = markers[closest_idx, :]

    for i in range(1, markers.shape[0]):
        d = euclid(markers[i, :], cd)
        if d < dist:
            closest_idx = i
            dist = d
            closest_coords = markers[i, :]
    if dist < 25:
        return closest_idx, closest_coords
    else:
        return None
    
def get_contour(arr, thickness):
    return cv2.dilate(arr, np.ones((2 * thickness + 1, 2 * thickness + 1))) - arr

def get_contour_points(arr):
    contours, hierarchy = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    shape = []
    for contour in contours[0]:
        shape.append(contour[0])
    return np.asarray(shape)

def get_max_pixel_id(cell_id, im_out, im_markers, im_pxs):
    im_cpy = im_out.copy()
    im_cpy[im_markers != cell_id] = 0
    ind = np.unravel_index(np.argmax(im_cpy), im_cpy.shape)
    del im_cpy
    return im_pxs[ind]

def get_cell_contour(cell_id, im_out, im_markers, im_pxs):
    contour_cell = np.zeros(im_markers.shape).astype('uint8')
    contour_cell[im_markers == cell_id] = 1
    shape = get_contour_points(contour_cell)
    p = Polygon(shape, edgecolor='r', closed=True, fill=None, lw=3)
    del contour_cell
    return p

def get_max_px_contour(cell_id, im_out, im_markers, im_pxs):
    contour_px = np.zeros(im_markers.shape).astype('uint8')
    contour_px[im_pxs == get_max_pixel_id(cell_id, im_out, im_markers, im_pxs)] = 1
    shape = get_contour_points(contour_px)
    p = Polygon(shape, edgecolor='g', closed=True, fill=None, lw=3)
    del contour_px
    return p

def get_cover_px(cell_id, im_markers, im_pxs):
    contour_cell = im_pxs.copy()
    contour_cell[im_markers != cell_id] = 0
    return contour_cell

def get_cover_px_contour(cell_id, im_out, im_markers, im_pxs):
    contour_cell = get_cover_px(cell_id, im_markers, im_pxs)
    shape = get_contour_points(np.isin(im_pxs, np.unique(contour_cell)[1:]).astype('uint8'))
    p = Polygon(shape, edgecolor='b', closed=True, fill=None, lw=3)
    del contour_cell, shape
    return p

def get_watershed_px_contour(cell_id, im_out, im_markers, im_watershed):
    contour_cell = get_cover_px(cell_id, im_watershed, im_watershed)
    
    if np.sum(contour_cell) == 0:
        return None
    
    shape = get_contour_points(contour_cell.astype('uint8'))
    p = Polygon(shape, edgecolor='y', closed=True, fill=None, lw=3)
    del contour_cell, shape
    return p

def is_adjacent(cell_id, im_markers, im_pxs):
    contour_cell = im_pxs.copy()
    contour_cell[im_markers != cell_id] = 0
    isin = np.isin(im_pxs, np.unique(contour_cell)[1:])
    covered_cells = im_markers.copy()
    covered_cells[np.logical_not(isin)] = 0
    cnt = np.unique(covered_cells)[1:]
    del contour_cell, covered_cells
    return cnt.tolist()

def get_max_px_signal_by_cell_id(cell_id, im_src, im_out, im_markers, im_pxs):
    px = get_max_pixel_id(cell_id, im_out, im_markers, im_pxs)
    coord = get_coord_from_idx(px)
    return im_src[:, coord[0], coord[1]]

def get_cover_px_well_by_cell_id(cell_id, im_src, im_markers, im_pxs):
    contour_cell = im_pxs.copy()
    contour_cell[im_markers != cell_id] = 0
    idxs = np.unravel_index(np.unique(contour_cell)[1:], (80,80))
    pxs = np.array([idxs[0], idxs[1]]).T
    masked = np.zeros(im_src.shape)
    for i in range(0, pxs.shape[0]):
        masked[: ,pxs[i, 0], pxs[i, 1]] = im_src[: ,pxs[i, 0], pxs[i, 1]]
    del contour_cell
    return masked

def get_cover_px_signal_by_cell_id(cell_id, im_src, im_out, im_markers, im_pxs):
    contour_cell = im_pxs.copy()
    contour_cell[im_markers != cell_id] = 0
    idxs = np.unravel_index(np.unique(contour_cell)[1:], (80,80))
    pxs = np.array([idxs[0], idxs[1]]).T
    signals = []
    for i in range(0, pxs.shape[0]):
        signals.append(im_src[: ,pxs[i, 0], pxs[i, 1]])
    signals = np.asarray(signals)
    int_signal = np.sum(signals, axis=0)
    del contour_cell
    return int_signal

def get_weighted_cover_px_signal_by_cell_id(cell_id, im_out, im_markers, im_pxs):
    signals = []
    contour_px = im_pxs.copy()
    contour_px[im_markers != cell_id] = 0
    for idx in np.unique(contour_px)[1:]:
        x, y = np.unravel_index(idx, (80,80))
        ratio = np.count_nonzero(contour_px == idx) / np.count_nonzero(im_pxs == idx)
        signals.append(im_out[:, x, y] * ratio)
    signals = np.asarray(signals)
    int_signal = np.sum(signals, axis=0)
    del contour_px
    return int_signal

def get_area_by_cell_id(cell_id, im_markers, px_size):
    signals = []
    contour_px = im_markers.copy()
    contour_px[im_markers != cell_id] = 0
    area = np.count_nonzero(contour_px) * px_size
    del contour_px
    return area