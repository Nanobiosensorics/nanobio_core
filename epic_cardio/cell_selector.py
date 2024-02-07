import matplotlib.pyplot as plt
import numpy as np
import math
from .math_ops import get_max_well

class Coordinates(list):
    def __init__(self):
        self._inner_list = list()

    def __len__(self):
        return len(self._inner_list)

    def __delitem__(self, index):
        self._inner_list.__delitem__(index)

    def __setitem__(self, index, value):
        self._inner_list.__setitem__(index, value)

    def __getitem__(self, index):
        return self._inner_list.__getitem__(index)

    def in_list(self, value):
        for item in self._inner_list:
            if item[0] == value[0] and item[1] == value[1]:
                return True
        return False

    def append(self, value):
        if not self.in_list(value):
            self._inner_list.append(value)
#         print(self._inner_list)

    def insert(self, index, value):
        if not self.in_list(value):
            self._inner_list.insert(index, value)
    
    def remove(self, value):
        self._inner_list = [ item for item in self._inner_list if item[0] != value[0] or item[1] != value[1]]
    
    def append_array(self, arr):
        for i in range(arr.shape[0]):
            self.append(arr[i, :])
            
    def pr(self):
        print(self._inner_list)

    def get_coord_arrays(self):
        xs = []
        ys = []
        for item in self._inner_list:
            xs.append(item[0])
            ys.append(item[1])
        return xs, ys

class CellSelector():
    def __init__(self, well, coords):
        # import matplotlib.pyplot as plt
        # import numpy as np
        # import math
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.well = well
        self.ax.set_title("Well")
        self.im = self.ax.imshow(self.well[-1, :, :],interpolation='none', vmin = 0, vmax = 2000)
        self.distance = math.ceil(self.well.shape[1] / 100)
        self.rdots, = self.ax.plot(coords[:, 0], coords[:, 1], 'ro', markersize=5)
        self.gdots, = self.ax.plot([], [], 'go', markersize=5)
        annotation = [str(n) for n in range(coords.shape[0])]
        for i, txt in enumerate(annotation):
            self.ax.annotate(txt, (coords[i, 0] + .5, coords[i, 1]), fontsize=7, c='w')
        self.coords = coords
        self.sel_coords = Coordinates()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self)
        # plt.show()
        
    def __call__(self, event):
        # import matplotlib.pyplot as plt
        # import numpy as np
        def euclid(a, b):
            return np.linalg.norm(a-b)
        dist = []
        coord = []
        i = 0
        for xi, yi in self.coords:
            d = euclid(np.asarray([event.xdata, event.ydata]),np.asarray([xi, yi]))
            if d <= self.distance:
                dist.append(d)
                coord.append(i)
            i+=1
        if len(dist) > 0:
            i = np.argmin(dist)
            if not self.sel_coords.in_list(self.coords[coord[i], :]):
                self.sel_coords.append(self.coords[coord[i], :])
            else:
                self.sel_coords.remove(self.coords[coord[i], :])
            self.gdots.set_data(self.sel_coords.get_coord_arrays())
            self.fig.canvas.draw()
        
    def get_selected_coords(self):
        # import numpy as np
        x, y = self.sel_coords.get_coord_arrays()
        return np.asarray((x, y)).T
    
    def export_well(self, filename):
        # import numpy as np
        np.save(filename, self.well)
        print(f'Well data exported to {filename}!')
        
    def export_coords(self, filename):
        # import numpy as np
        np.save(filename, self.get_selected_coords())
        print(f'Selected coordinates exported to {filename}!')
    
    def export_data(self, w_filename, c_filename):
        self.export_well(w_filename)
        self.export_coords(c_filename)
        
class WellLineSelector:
    # Jelkiválasztó
    # Kézzel végig lehet futni a szelektált jeleken és meg lehet
    # adni, hogy melyikeket exportálja.
    # A jelek közötti navigáció lehetségesa billentyűzeten a balra, jobbra nyilakkal és
    # a mentés az ENTER billentyűvel
    def __init__(self, well, coords, lines, w_name=""):
        self.saved_ids = []
        self._lines_selected = []
        self._i = 1
        self._frame = well[-1, :, :].copy()
        self._well = well
        self._pts_arr = np.asarray(coords)
        self._lines_arr = np.asarray(lines)
        self._fig, (self._ax1,self._ax2) = plt.subplots(1,2, figsize=(16, 8))
        self._ax2.set_ylim((np.min(self._lines_arr), np.max(self._lines_arr)))

        self._im = self._ax1.imshow(self._frame, vmin = 0, vmax = 0.6)
        self._elm, = self._ax2.plot(np.linspace(0, self._lines_arr.shape[1], self._lines_arr.shape[1]), self._lines_arr[0, :])
        self._dots, = self._ax1.plot(self._pts_arr[0, 0], self._pts_arr[0, 1] , 'ro', markersize=5)
        self._selected, = self._ax1.plot(self._pts_arr[self.saved_ids, 0], self._pts_arr[self.saved_ids, 0] , 'go', markersize=5)
        self._fig.canvas.mpl_connect('key_press_event', self.on_press)
        self._ax1.set_title(w_name)
        self.draw_plot(0)

    def draw_plot(self, cell_id):
        self._elm.set_data((np.linspace(0, self._lines_arr.shape[1], self._lines_arr.shape[1]), self._lines_arr[cell_id, :]))
        self._ax2.set_title(f'Record: {cell_id}/{self._lines_arr.shape[0]}')
        self._im.set_data(self._well[np.argmax(self._lines_arr[cell_id, :]), :, :])
        self._dots.set_data((self._pts_arr[cell_id, 0], self._pts_arr[cell_id, 1]))
        self._selected.set_data((self._pts_arr[self.saved_ids, 0], self._pts_arr[self.saved_ids, 1]))
        self._fig.canvas.draw()

    def on_button_plus_clicked(self, b):
        self._i = self._i + 1 if self._i + 1 < self._lines_arr.shape[0] else self._lines_arr.shape[0]
        self.draw_plot(self._i - 1)
        if self._i == self._lines_arr.shape[0]:
            plt.close(self._fig)
        
        
    def on_button_minus_clicked(self, b):
        self._i = self._i - 1 if self._i - 1 > 0 else 1
        self.draw_plot(self._i - 1)
        
    def on_button_save_clicked(self, b):
        if self._i - 1 not in self.saved_ids:
            # lines_selected.append(lines_arr[i[0] - 1, :])
            self.saved_ids.append(self._i - 1)
        self.on_button_plus_clicked(b)
        
    def on_press(self, event):
        if event.key == 'right' or event.key == '6':
            self.on_button_plus_clicked(None)
        elif event.key == 'left' or event.key == '4':
            self.on_button_minus_clicked(None)
        elif event.key == 'enter':
            self.on_button_save_clicked(None)

class WellArrayLineSelector:
    # Jelkiválasztó
    # Kézzel végig lehet futni a szelektált jeleken és meg lehet
    # adni, hogy melyikeket exportálja.
    # A jelek közötti navigáció lehetségesa billentyűzeten a balra, jobbra nyilakkal és
    # a mentés az ENTER billentyűvel
    _ids = [ 'A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3', 'C4']
    def __init__(self, wells_data, times, phases, block=True):
        self.saved_ids = {name:[] for name in self._ids}
        self.closed = False
        self._well_id = 0
        self._well_max = 0
        self._wells_data = wells_data
        self._times = times
        self._phases = phases
        self.texts = []
        
        self._fig, (self._ax1,self._ax2) = plt.subplots(1,2, figsize=(16, 8))
        
        self.change_well()
        
        self._dots, = self._ax1.plot(self._pts_arr[0, 0], self._pts_arr[0, 1] , 'ro', markersize=5)
        self._selected, = self._ax1.plot(self._pts_arr[self.saved_ids[self._ids[self._well_id]], 0], self._pts_arr[self.saved_ids[self._ids[self._well_id]], 0] , 'go', markersize=5)
        self._ax1.set_xlabel('Pixel')
        self._ax1.set_ylabel('Pixel')
        self._ax2.set_xlabel('Time(s)')
        self._ax2.set_ylabel('WS(pm)')
        self._fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.draw_plot(0)
        plt.show(block=block)
        
    def _get_line(self, cell_id):
        current_line = self._wells_data[self._ids[self._well_id]][0][:, self._pts_arr[cell_id, 1], self._pts_arr[cell_id, 0]].copy()
        
        if len(self._phases) > 0:
            for p in self._phases:
                current_line[p] = np.nan
        return current_line
    
    def change_well(self):
        if self._well_id == len(self._ids):
            plt.close(self._fig)
            self.closed = True
        else:
            while(self._wells_data[self._ids[self._well_id]][1].shape[0] == 0):
                self._well_id += 1
            self._well = np.max(self._wells_data[self._ids[self._well_id]][0], axis=0)
            self._pts_arr = np.asarray(self._wells_data[self._ids[self._well_id]][1])
            
            self._i = 1
            if len(self.texts) > 0:
                for txt in self.texts:
                    txt.remove()
                self.texts.clear()
                
            if hasattr(self, '_im'):
                self._im.remove()
            if hasattr(self, '_elm'):
                self._elm.remove()
            
            current_line = self._get_line(0)
                    
            self._im = self._ax1.imshow(self._well, vmin = 0, vmax=np.max(self._well))
            self._elm, = self._ax2.plot(self._times[:len(current_line)], current_line)
        
            self._well_max = max([np.nanmax(self._get_line(n)) for n in range(self._pts_arr.shape[0])])

    def draw_plot(self, cell_id):
        current_line = self._get_line(cell_id)
        self._elm.set_data(self._times[:len(current_line)], current_line)
        self._ax1.set_title(self._ids[self._well_id])
        self._ax2.set_title(f'Record: {cell_id + 1}/{self._pts_arr.shape[0]}')
        self._dots.set_data((self._pts_arr[cell_id, 0], self._pts_arr[cell_id, 1]))
        self._selected.set_data((self._pts_arr[self.saved_ids[self._ids[self._well_id]], 0], self._pts_arr[self.saved_ids[self._ids[self._well_id]], 1]))
        self._ax2.set_ylim(-100, self._well_max)
        self._fig.canvas.draw()

    def on_button_plus_clicked(self, b):
        self._i += 1
        if self._i >= self._pts_arr.shape[0] + 1:
            self._well_id += 1
            self.change_well()
        if self.closed is False:
            self.draw_plot(self._i - 1)
        
    def on_button_minus_clicked(self, b):
        self._i = self._i - 1 if self._i - 1 > 0 else 1
        self.draw_plot(self._i - 1)
        
    def on_button_save_clicked(self, b):
        if self._i - 1 not in self.saved_ids:
            # lines_selected.append(lines_arr[i[0] - 1, :])
            self.saved_ids[self._ids[self._well_id]].append(self._i - 1)
            txt = self._ax1.text(self._pts_arr[self._i - 1, 0] + 1, self._pts_arr[self._i - 1, 1], f"{len(self.saved_ids[self._ids[self._well_id]]) - 1}", color='white')
            self.texts.append(txt)
        self.on_button_plus_clicked(b)
        
    def on_press(self, event):
        if event.key == 'right' or event.key == '6':
            self.on_button_plus_clicked(None)
        elif event.key == 'left' or event.key == '4':
            self.on_button_minus_clicked(None)
        elif event.key == 'down':
            self._well_id += 1
            self.change_well()
            if self.closed is False:
                self.draw_plot(self._i - 1)
        elif event.key == 'enter':
            self.on_button_save_clicked(None)

class SignalCutter:
    def __init__(self, signals):
        self._signals = signals
        self._fig, self._ax = plt.subplots(1,1,figsize=(20, 10))
        for i in range(signals.shape[0]):
            self._ax.plot(signals[i, :])
        self._cut_id = 0
        self._line = self._ax.axvline(10, color='gray', linestyle='--')

        self._fig.canvas.mpl_connect("motion_notify_event", self._hover)
        self._fig.canvas.mpl_connect("button_press_event", self._push)
        plt.show()
    
    def _hover(self, evt):
        if evt.inaxes is not None:
            self._line.set_xdata(evt.xdata)
            self._fig.canvas.draw()

    def _push(self, evt):
        if evt.xdata is not None:
            self._cut_id = round(evt.xdata)
            plt.close(self._fig)
    
    def get_cut_id(self):
        return self._cut_id