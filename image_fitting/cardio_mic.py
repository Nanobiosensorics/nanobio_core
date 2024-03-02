import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from scipy.spatial.distance import euclidean
import json


class CardioMicScaling:
    MIC_5X = 2134 / 2.22
    MIC_10X = 2134 * 0.9817
    MIC_20X = 2134 * 1.81

class CardioMicFitter:
    def __init__(self, well, mic, result_path, points=[], scaling=CardioMicScaling.MIC_5X, block=True, save_params=False, load_params=None, name=None, show_scatter=False):
        self.name = name
        self.closed = False
        self.tuned = False
        self.save_params = save_params
        self.cardio_points = []
        self.mic_points = []
        self._well_id = 0
        self.distance = 100
        self.result_path = result_path
        self.translation = np.array([1420, 955])
        self.scale, _ = CardioMicFitter._get_scale(scaling)
        self.points = np.asarray(points)
        self.show_scatter = show_scatter
        
        if load_params != None:
            if os.path.exists(load_params) and load_params.endswith("json"):
                with open(load_params, "r") as fp:
                    obj = json.load(fp)
                    self.scale = obj["scale"]
                    self.translation = np.array([obj["t_0"], obj["t_1"]])
        
        img = well.copy()
        
        if len(img.shape) > 2:
            img = np.max(img, axis=0)
            
        self.max_well = img
        self._mic, self._well = mic, cv2.resize(self.max_well, (self.scale, self.scale), interpolation=cv2.INTER_NEAREST)
        
        self._fig, self._ax = plt.subplots(figsize=(16, 8))
        self._ax.set_axis_off()
        self._im = self._ax.imshow(self._mic, cmap='gray', vmin = 0, vmax = np.max(self._well))
        self._elm = self._ax.imshow(self._well, alpha=.4,
                    extent = [self.translation[0], self.translation[0]  + self._well.shape[0],
                                               self.translation[1] + self._well.shape[1], self.translation[1]])
        self.pts1 = None
        self.pts2 = None
        self._fig.canvas.mpl_connect('key_press_event', self.on_press)
        self._fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.make_title()
        self.draw_plot()
        plt.show(block=block)
    
    @staticmethod
    def _get_scale(scale):
        MIC_PX_PER_UM = scale / 1000
        EPIC_PX_PER_UM = 1/25
        EPIC_CARDIO_SCALE = int(80 * (MIC_PX_PER_UM / EPIC_PX_PER_UM))
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
        print(self.points)
        if len(self.points) > 0:
            pixel_conv = self.scale / 80
            self.points = (self.points + .5) * pixel_conv + self.translation
            if self.show_scatter:
                self._ax.scatter(self.points[:, 0], self.points[:, 1], s = int(pixel_conv / 4), c = 'r')
            plt.rcParams.update({'font.size': 10})
            for i in range(self.points.shape[0]):
                self._ax.text(self.points[i, 0] + pixel_conv, self.points[i, 1] - pixel_conv, f"{i}", color="white")
            self._fig.canvas.draw()
            
        self._fig.savefig(os.path.join(self.result_path, f'{"well" if self.name == None else self.name}_cardio_microscope.png'), bbox_inches='tight', pad_inches=0)
        if self.save_params:
            obj = {
                "scale": self.scale,
                "t_0": self.translation[0],
                "t_1": self.translation[1],
            }
            save_path = os.path.join(self.result_path, "metadata")
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(os.path.join(save_path, f'{"well" if self.name == None else self.name}_map_params.json')) as fp:
                json.dump(obj, fp)
        
        plt.close(self._fig)
        self.closed = True
    
    def _remove_coordinates(self):
        if self.pts1 is not None:
            self.pts1.remove()
            self.pts1 = None
        if self.pts2 is not None:
            self.pts2.remove()
            self.pts2 = None
        
    def handle_key_press(self, evt):
        if evt.key == 'r':
            self.tuned = not self.tuned
            self.cardio_points = []
            self.mic_points = []
            self._remove_coordinates()
            self.draw_plot()   
        elif self.tuned:
            if evt.key == 'm' and hasattr(evt, 'button'):
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
                elif evt.key in '12345678':
                    if evt.key == '1':
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
                    self.make_title()
                    self._fig.canvas.draw()
                elif evt.key == 'n':
                    self.next_pressed()        
                elif evt.key == 'enter':
                    self.enter_pressed()
        else:
            if hasattr(evt, 'button'):
                if evt.xdata != None and evt.ydata != None:
                    if evt.key == 'a':
                        self.cardio_points.append((evt.xdata, evt.ydata))
                        if self.pts1 == None:
                            self.pts1, = self._ax.plot([l[0] for l in self.cardio_points], [l[1] for l in self.cardio_points], 'bo')
                        else:
                            self.pts1.set_xdata([l[0] for l in self.cardio_points])
                            self.pts1.set_ydata([l[1] for l in self.cardio_points])
                        self._fig.canvas.draw()
                    elif evt.key == 'd':
                        self.mic_points.append((evt.xdata, evt.ydata))
                        if self.pts2 == None:
                            self.pts2, = self._ax.plot([l[0] for l in self.mic_points], [l[1] for l in self.mic_points], 'ro')
                        else:
                            self.pts2.set_xdata([l[0] for l in self.mic_points])
                            self.pts2.set_ydata([l[1] for l in self.mic_points])
                        self._fig.canvas.draw()
            elif evt.key == 't':
                # sc = euclidean(self.mic_points[1][0], self.mic_points[1][1]) / euclidean(self.cardio_points[1][0], self.cardio_points[1][1])
                self.cardio_points = np.asarray(self.cardio_points)
                self.mic_points = np.asarray(self.mic_points)
                distances = []
                for i in range(0, len(self.cardio_points)):
                    for j in range(i + 1, len(self.cardio_points)):
                        distances.append(euclidean(self.mic_points[i], self.mic_points[j]) / 
                                         euclidean(self.cardio_points[i], self.cardio_points[j]))
                sc = np.mean(distances)
                self.scale *= sc
                self.scale = int(self.scale)
                self._well = cv2.resize(self.max_well, (self.scale, self.scale), interpolation=cv2.INTER_NEAREST)
                dist = np.mean(np.asarray([self.mic_points[i] - sc * self.cardio_points[i] for i in range(len(self.cardio_points))]), axis=0)
                self.translation[0] += dist[0]
                self.translation[1] += dist[1]
                self.tuned = True
                self._remove_coordinates()
                self.draw_plot()                     
        
    def on_press(self, evt):
        self.handle_key_press(evt)
        
class CardioMicFitterMultipleWell(CardioMicFitter):
    _ids = [ 'A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3', 'C4']
    def __init__(self, wells_data, mics_data, result_path, scaling=CardioMicScaling.MIC_5X, block=True, retune=False):
        self._well_id = 0
        self._wells_data = wells_data
        self._mics_data = mics_data
        self.translations = []
        self.retune = retune
        
        super().__init__(self._wells_data[self._ids[self._well_id]], self._mics_data[self._ids[self._well_id]], 
                         result_path, scaling=scaling, block=block)
    
    def plot_well(self):
        if self._well_id == len(self._ids):
            plt.close(self._fig)
            self.closed = True
        else:        
            img = self._wells_data[self._ids[self._well_id]].copy()
            
            if len(img.shape) > 2:
                img = np.max(img, axis=0)
                
            self._well = cv2.resize(img, (self.scale, self.scale), interpolation=cv2.INTER_NEAREST)
            self._mic = self._mics_data[self._ids[self._well_id]]
            if hasattr(self, '_im'):
                self._im.set_data(self._mic)
            if hasattr(self, '_elm'):
                self._elm.remove()
                self._elm = self._ax.imshow(self._well, alpha=.4,
                    extent = [self.translation[0], self.translation[0]  + self._well.shape[0],
                                               self.translation[1] + self._well.shape[1], self.translation[1]])
                
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
            if self.retune:
                self._remove_coordinates()
                self.tuned = False
            self.plot_well()
            self.draw_plot() 
            
    def enter_pressed(self):
        self._ax.set_title(None)
        self._fig.savefig(os.path.join(self.result_path, self._ids[self._well_id] + '_cardio_microscope.png'), bbox_inches='tight', pad_inches=0)
        self.translations.append(self.translation.copy())
        self._well_id += 1
        
        if not self.check_close():
            if self.retune:
                self._remove_coordinates()
                self.tuned = False
            self.plot_well()
            self.draw_plot() 
