import numpy as np
import matplotlib.pyplot as plt

from .preview_scene import WellPreviewSceneMixin


class WellArrayBackgroundSelector(WellPreviewSceneMixin):
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
        self._init_preview_state()
        self._dots = None
        self._im = None 
        self.selected_coords = {}
        for idx in WellArrayBackgroundSelector._ids:
            if idx in list(coords.keys()):
                self.selected_coords[idx] = list(coords[idx])
            else:
                self.selected_coords[idx] = []
        self._fig = plt.figure(figsize=(8, 8))
        self._build_main_layout()
        self._fig.canvas.mpl_connect('key_press_event', self.on_press)
        self._fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.change_well()
        self.draw_plot()
        plt.show(block=block)

    def _build_main_layout(self):
        self._fig.clf()
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._ax.set_xlabel('Pixel')
        self._ax.set_ylabel('Pixel')
        self._dots = None
        self._im = None

    def _redraw_main_scene_on_preview_exit(self):
        self.change_well()
        self.draw_plot()

    def _preview_well_image(self, well_name):
        return np.max(self._wells_data[well_name], axis=0)

    def _draw_preview_overlay(self, ax, well_name):
        arr = self.selected_coords[well_name]
        if len(arr) > 0:
            ax.plot([e[0] for e in arr], [e[1] for e in arr], 'ro', markersize=2)
    
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

    def _handle_global_key(self, event):
        key = getattr(event, 'key', None)
        if key == 'f':
            self.closed = True
            plt.close(self._fig)
            return True
        if key == 'a':
            self._toggle_preview_mode()
            return True
        return False

    def _handle_preview_event(self, event):
        if hasattr(event, 'button'):
            self._handle_preview_click(event)

    def _handle_main_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        self.selected_coords[self._ids[self._well_id]].append((round(event.xdata),round(event.ydata)))
        self.draw_plot()

    def _handle_main_key(self, event):
        if event.key == 'right' or event.key == '6':
            self.on_button_plus_clicked(None)
        elif event.key == 'left' or event.key == '4':
            self.on_button_minus_clicked(None)
        elif event.key == 'enter':
            self.on_button_save_clicked(None)
        elif event.key == 'delete' or event.key == 'backspace':
            if len(self.selected_coords[self._ids[self._well_id]]) > 0:
                self.selected_coords[self._ids[self._well_id]].pop()
                self.draw_plot()

    def on_press(self, event):
        if self._handle_global_key(event):
            return
        if self._preview_mode:
            self._handle_preview_event(event)
            return
        if hasattr(event, 'button'):
            self._handle_main_click(event)
            return
        self._handle_main_key(event)
