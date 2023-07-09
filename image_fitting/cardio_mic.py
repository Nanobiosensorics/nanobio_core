import matplotlib.pyplot as plt
from ..epic_cardio.math_ops import get_max_well
import numpy as np
import cv2
import os

class CardioMicScaling:
    MIC_5X = 2134 / 2.22
    MIC_20X = 2134 * 1.81

class CardioMicFitter:
    def __init__(self, well, mic, result_path, scaling=CardioMicScaling.MIC_5X, block=True):
        self.closed = False
        self._well_id = 0
        self.translation = np.array([1420, 955])
        self.distance = 100
        self.result_path = result_path
        self.scale, _ = self._get_scale(scaling)
        self._mic, self._well = mic, cv2.resize(get_max_well(well), (self.scale, self.scale), interpolation=cv2.INTER_NEAREST)
        
        self._fig, self._ax = plt.subplots(figsize=(16, 8))
        self._ax.set_axis_off()
        self._im = self._ax.imshow(self._mic, cmap='gray') # , vmin = np.min(self._well), vmax = np.max(self._well)
        self._elm = self._ax.imshow(self._well, alpha=.6,
                    extent = [self.translation[0], self.translation[0]  + self._well.shape[0],
                                               self.translation[1] + self._well.shape[1], self.translation[1]],)
        self._fig.canvas.mpl_connect('key_press_event', self.on_press)
        self._fig.canvas.mpl_connect('button_press_event', self.on_press)
        self._ax.set_title(f'Current translation {self.translation}, speed {self.distance}')
        self.draw_plot()
        plt.show(block=block)
        
    def _get_scale(self, scaling):
        MIC_PX_PER_UM = scaling / 1000
        EPIC_PX_PER_UM = 1/25
        EPIC_CARDIO_SCALE = int(80 * (MIC_PX_PER_UM / EPIC_PX_PER_UM) * 0.978)
        MIC_UM_PER_PX = 1 / MIC_PX_PER_UM
        MIC_PX_AREA = MIC_UM_PER_PX**2
        return EPIC_CARDIO_SCALE, MIC_UM_PER_PX

    def draw_plot(self):
        self._elm.set_extent([self.translation[0], self.translation[0]  + self._well.shape[0],
                        self.translation[1] + self._well.shape[1], self.translation[1]])
        self._ax.set_xlim((0, self._mic.shape[1]))
        self._ax.set_ylim((self._mic.shape[0], 0))
        self._fig.canvas.draw()
        
    def make_title(self):
        self._ax.set_title(f'Current translation {self.translation}, speed {self.distance}')
        
    def next_pressed(self):
        plt.close(self._fig)
        self.closed = True
            
    def enter_pressed(self):
        self._ax.set_title(None)
        self._fig.savefig(os.path.join(self.result_path, 'well_cardio_microscope.png'), bbox_inches='tight', pad_inches=0)
        plt.close(self._fig)
        self.closed = True
        
    def handle_key_press(self, evt):
        if hasattr(evt, 'button'):
            if evt.xdata != None and evt.ydata != None:
                self.translation[0] = round(evt.xdata)
                self.translation[1] = round(evt.ydata)
                self.draw_plot()
        elif evt.key != None:
            if evt.key == 'right':
                self.translation[0] = self.translation[0] + self.distance # if translation[0] + distance < im_dst.shape[1] else im_dst.shape[1]
                self.draw_plot()
            elif evt.key == 'left':
                self.translation[0] = self.translation[0] - self.distance # if translation[0] - distance > 0 else 0
                self.draw_plot()
            elif evt.key == 'down':
                self.translation[1] = self.translation[1] + self.distance # if translation[1] + distance < im_dst.shape[0] else im_dst.shape[0]
                self.draw_plot()
            elif evt.key == 'up':
                self.translation[1] = self.translation[1] - self.distance # if translation[1] - distance > 0 else 0
                self.draw_plot()
            elif evt.key == '1':
                self.distance = 1
            elif evt.key == '2':
                self.distance = 5
            elif evt.key == '3':
                self.distance = 10
            elif evt.key == '4':
                self.distance = 20
            elif evt.key == '5':
                self.distance = 50
            elif evt.key == '6':
                self.distance = 100
            elif evt.key == '7':
                self.distance = 200
            elif evt.key == '8':
                self.distance = 500
            elif evt.key == 'n':
                self.next_pressed()        
            elif evt.key == 'enter':
                self.enter_pressed()
        
    def on_press(self, evt):
        self.handle_key_press(evt)
        self.make_title()
        self._fig.canvas.draw()
        
class CardioMicFitterMultipleWell(CardioMicFitter):
    _ids = [ 'A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3', 'C4']
    def __init__(self, wells_data, mics_data, result_path, scaling=CardioMicScaling.MIC_5X, block=True):
        self._well_id = 0
        self._wells_data = wells_data
        self._mics_data = mics_data
        self.translations = []
        
        super().__init__(self._wells_data[self._ids[self._well_id]], self._mics_data[self._ids[self._well_id]], 
                         result_path, scaling=scaling, block=block)
    
    def plot_well(self):
        if self._well_id == len(self._ids):
            plt.close(self._fig)
            self.closed = True
        else:
            self._well = cv2.resize(get_max_well(self._wells_data[self._ids[self._well_id]]), (self.scale, self.scale), interpolation=cv2.INTER_NEAREST)
            self._mic = self._mics_data[self._ids[self._well_id]]
            if hasattr(self, '_im'):
                self._im.set_data(self._mic)
            if hasattr(self, '_elm'):
                self._elm.set_data(self._well)
                
    def make_title(self):
        self._ax.set_title(f'Well {self._ids[self._well_id]}, Current translation {self.translation}, speed {self.distance}')
                
    def check_close(self):
        if self._well_id == len(self._ids):
            plt.close(self._fig)
            self.closed = True
        return self.closed
                
    def next_pressed(self):
        self._well_id += 1
        if not self.check_close():
            self.plot_well()
            self.draw_plot() 
            
    def enter_pressed(self):
        self._ax.set_title(None)
        self._fig.savefig(os.path.join(self.result_path, self._ids[self._well_id] + '_cardio_microscope.png'), bbox_inches='tight', pad_inches=0)
        self.translations.append(self.translation.copy())
        self._well_id += 1
        
        if not self.check_close():
            self.plot_well()
            self.draw_plot() 