import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

def corr_data(data):
    corr_data = data.copy()
    for n in range(0, corr_data.shape[0]):
        corr_data[n, :] -= corr_data[n, 1] - corr_data[n, 0]
    for n in range(0, corr_data.shape[0]):
        corr_data[n, :] -= corr_data[n, 0]
    return corr_data

def select_indices(array, threshold, num_indices, spacing = 2, random_distance=5, layer=0):
    # Get the indices of the true elements in the boolean array
    
    bool_array = array < (threshold + layer * 25)
    
    true_indices = np.argwhere(bool_array)
    dst = distance_transform_edt(bool_array)

    rows, cols = bool_array.shape
    center_row = rows // 2
    center_col = cols // 2
    
    distances_to_center = np.array([np.sqrt((y - center_row)**2 + (x - center_col)**2) for y, x in true_indices])
    preference_weights = 1 / (distances_to_center + 1)  # Adding 1 to avoid division by zero
    
    chosen_indices = []
    
    n_attempts = 0
    
    while len(chosen_indices) < num_indices and n_attempts < 1000:
        # Choose a random true index
        idx = np.random.choice(len(true_indices), p=preference_weights.ravel() / np.sum(preference_weights))
        random_true_index = np.unravel_index(idx, (rows, cols))
        
        # Check if the chosen index is at least 'random_distance' units away from other chosen indices
        if all(np.linalg.norm(np.array(chosen_index) - random_true_index) >= random_distance for chosen_index in chosen_indices) and dst[random_true_index[0],random_true_index[1]] >= spacing:
            chosen_indices.append(random_true_index)
            
        n_attempts += 1
    if len(chosen_indices) < num_indices:
        if layer < 10:
            return select_indices(array, threshold, num_indices, spacing, random_distance, layer + 1)
        else:
            if len(chosen_indices) > 0:
                x_indices, y_indices = zip(*chosen_indices)
                return x_indices, y_indices
            return []
        
    # Split the chosen indices into x and y groups
    x_indices, y_indices = zip(*chosen_indices)
    
    return x_indices, y_indices

def dilate_mask(mask, times=1):
    for i in range(times):
        mask = ndimage.binary_dilation(mask, [
        [False, True, False],
        [ True, True,  True],
        [False, True, False],
    ])
    return mask

def correct_well(well, threshold = 75, coords=[], mode='mean'):
    corr_data = well.copy()
    
    df = np.abs(np.diff(corr_data, axis=0))
    df = np.sum(df, axis=0)
    mask = np.logical_or(df == 0, df > 1000)
    mask = dilate_mask(mask, 1)
    
    corr_data -= corr_data[0, :, :]
    corr_data *= 1000

    if len(coords) > 0:
        coords = [[e[0] for e in coords], [e[1] for e in coords]]
    else:
        coords = select_indices(corr_data[-1], threshold, 7, 2, 2)

    if len(coords) > 0:
        filter_method = np.median if mode == 'median' else np.mean
        fltr = np.transpose(np.tile(filter_method(corr_data[:, coords[1], coords[0]], axis=1), (80, 80, 1)), (2,0,1))
        corr_data -= fltr
        # corr_data[:, mask] = 0
    else:
        print('Could not perform random background correction!')
    corr_data[np.abs(corr_data) > 5000] = 0
    corr_data[:, mask] = 0
    
    return corr_data, {} if len(coords) == 0 else list(zip(coords[0], coords[1])), mask

class WellArrayBackgroundSelector:
    # Jelkiválasztó
    # Kézzel végig lehet futni a szelektált jeleken és meg lehet
    # adni, hogy melyikeket exportálja.
    # A jelek közötti navigáció lehetségesa billentyűzeten a balra, jobbra nyilakkal és
    # a mentés az ENTER billentyűvel
    _ids = [ 'A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3', 'C4']
    def __init__(self, wells_data, coords={}, block=True):
        self.saved_ids = {name:[] for name in self._ids}
        self.closed = False
        self._well_id = 0
        self._wells_data = wells_data
        self._dots = None
        self._im = None 
        self.selected_coords = {}
        for idx in WellArrayBackgroundSelector._ids:
            if idx in list(coords.keys()):
                self.selected_coords[idx] = list(coords[idx])
            else:
                self.selected_coords[idx] = []
        self._fig, self._ax = plt.subplots(1, figsize=(8, 8))
        self._ax.set_xlabel('Pixel')
        self._ax.set_ylabel('Pixel')
        self._fig.canvas.mpl_connect('key_press_event', self.on_press)
        self._fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.change_well()
        self.draw_plot()
        plt.show(block=block)
    
    def change_well(self):
        if self._well_id == len(self._ids):
            plt.close(self._fig)
            self.closed = True
        else:
            self._well = np.max(self._wells_data[self._ids[self._well_id]], axis = 0)

            if self._well_id != 0 and len(self.selected_coords[self._ids[self._well_id]]) == 0:
                self.selected_coords[self._ids[self._well_id]] = self.selected_coords[self._ids[self._well_id - 1]].copy()

            if self._dots != None:
                self._dots.remove()
                self._dots = None
            

    def draw_plot(self):
        if not self.closed:
            self._ax.set_title(self._ids[self._well_id])
            
            if self._im != None:
                self._im.remove()
            self._im = self._ax.imshow(self._well, vmin = 0, vmax=np.max(self._well))
            arr = self.selected_coords[self._ids[self._well_id]]
            if len(arr) > 0:
                if self._dots == None:
                    self._dots, = self._ax.plot([e[0] for e in arr], [e[1] for e in arr], 'ro', markersize=5)
                else:
                    self._dots.set_xdata([e[0] for e in arr])
                    self._dots.set_ydata([e[1] for e in arr])
            elif self._dots != None:
                self._dots.remove()
                self._dots = None
            self._fig.canvas.draw()

    def on_button_plus_clicked(self, b):
        if self._well_id < len(self._ids):
            self._well_id += 1
            self.change_well()
            self.draw_plot()
        
    def on_button_minus_clicked(self, b):
        if self._well_id > 0:
            self._well_id -= 1
            self.change_well()
            self.draw_plot()
        
    def on_button_save_clicked(self, b):
        self._well_id += 1
        self.change_well()
        self.draw_plot()

    def on_press(self, event):
        if hasattr(event, 'button'):
            if event.xdata is None or event.ydata is None:
                pass
            self.selected_coords[self._ids[self._well_id]].append((round(event.xdata),round(event.ydata)))
            self.draw_plot()
        elif event.key == 'right' or event.key == '6':
            self.on_button_plus_clicked(None)
        elif event.key == 'left' or event.key == '4':
            self.on_button_minus_clicked(None)
        elif event.key == 'enter':
            self.on_button_save_clicked(None)
        elif event.key == 'delete' or event.key == 'backspace':
            if len(self.selected_coords[self._ids[self._well_id]]) > 0:
                self.selected_coords[self._ids[self._well_id]].pop()
                self.draw_plot()