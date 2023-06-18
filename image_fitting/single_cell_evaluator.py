import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import numpy as np
from .funcs import *

class SingleCellDisplayContour:
    CELL = 0,
    MAX = 1,
    COVER = 2,
    ALL = 255

class CardioMicSingleCellEvaluator():
    
    def __init__(self, well, im_cardio, im_mic, im_markers, im_pxs, im_contour, markers, centers, px_size, resolution = 1, 
                 display_contours = [
                    SingleCellDisplayContour.ALL,   
                 ]):
        self.disp = display_contours
        self.idx = 0
        self.well = well
        self.im_cardio = im_cardio
        self.im_mic = im_mic
        self.im_markers = im_markers
        self.im_pxs = im_pxs
        self.im_contour = im_contour
        self.markers = markers
        self.centers = centers
        self.resolution = resolution
        self.px_size = px_size
        self.selected_coords = []
        # button_plus = widgets.Button(description="⮞")
        # button_minus = widgets.Button(description="⮜")
        # slider = widgets.IntSlider(value=1, min=1, max=len(self.centers) - 1)
        # output = widgets.Output()
        # home = widgets.Button(description="Home")
        # box = widgets.HBox([button_minus, button_plus])
        # display(box)

        self.fig, self.ax = plt.subplots(2, 2, figsize=(16, 12))
        self.ax_mic = self.ax[0, 0]
        self.ax_cell = self.ax[1, 0]
        self.ax_max = self.ax[0, 1]
        self.ax_int = self.ax[1, 1]
        self.ax_mic.axes.get_xaxis().set_visible(False)
        self.ax_mic.axes.get_yaxis().set_visible(False)
        self.ax_cell.axes.get_xaxis().set_visible(False)
        self.ax_cell.axes.get_yaxis().set_visible(False)
        # ax[2, 0].set_visible(False)
        # ax[2, 1].set_visible(False)
        self.ax_mic.set_position([0, 0.525, .45, .45])
        self.ax_cell.set_position([0, 0.025, .45, .45])
        # ax4.set_visible(True)
        # ax.set_position([0, 0, .5, .5])
        # ax2.set_position([0, .25, .5, .5])

        self.ax_mic.imshow(self.im_mic)
        self.ax_mic.imshow(self.im_cardio, alpha=.4, 
                        vmin = np.mean(self.im_cardio) - 3*np.std(self.im_cardio), 
                        vmax = np.mean(self.im_cardio) + 3*np.std(self.im_cardio))
        self.ax_cell.imshow(self.im_mic)
        self.ax_cell.imshow(self.im_cardio, alpha=.4, 
                        vmin = np.mean(self.im_cardio) - 3*np.std(self.im_cardio), 
                        vmax = np.mean(self.im_cardio) + 3*np.std(self.im_cardio))
        self.ax_max.set_ylim((-.05, 2500))
        self.ax_int.set_ylim((np.min(self.well), np.max(self.well)))
        self.ax_mic.set_title('Cells')
        self.ax_max.set_title('Max pixel signal')
        self.ax_int.set_title('Integrated pixels signal')
        # ax[2, 1].set_title('Integrated weighted pixels signal')
        self.disp_pts, = self.ax_mic.plot(self.centers[:, 0], self.centers[:, 1], 'bo', markersize=3)
        self.crnt_pt, = self.ax_mic.plot(self.centers[0, 0],self.centers[0, 1], 'bo', markersize=10, mfc='none')
        self.selected_pts, =  self.ax_mic.plot( [], [], 'ro', markersize=3)
        # elm2 = ax.imshow(contour_img_display, alpha=.2)
        # elm3 = ax[1, 0].imshow(contour_img_display, alpha=.2)
        self.line_max, = self.ax_max.plot(np.linspace(0, self.well.shape[0], self.well.shape[0]), self.well[:, 0, 0])
        self.line_int, = self.ax_int.plot(np.linspace(0, self.well.shape[0], self.well.shape[0]), self.well[:, 0, 0])
        # elm6, = ax[2, 1].plot(np.linspace(0, self.well.shape[0], self.well.shape[0]), self.well[:, 0, 0])
        
        # button_plus.on_click(on_button_plus_clicked)
        # button_minus.on_click(on_button_minus_clicked)
        # slider.observe(slider_change, names='value')
        # home.on_click(home_onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_button_press)

        self.draw_plot()

    def change_ax_limit(self, x_center, y_center):
        self.ax_cell.set_xlim((x_center - 100 * self.resolution, x_center + 100 * self.resolution))
        self.ax_cell.set_ylim((y_center + 100 * self.resolution, y_center - 100 * self.resolution))
        self.crnt_pt.set_data(((x_center),(y_center)))

    def draw_plot(self):
        self.ax_cell.patches = []
        self.ax_cell.set_title(f'Cell {self.idx + 1}/{len(self.markers)}, Area {np.round(get_area_by_cell_id(self.markers[self.idx], self.im_markers, self.px_size), 2)} μm²')
        self.change_ax_limit(self.centers[self.idx, 0], self.centers[self.idx, 1])
        if SingleCellDisplayContour.CELL in self.disp or SingleCellDisplayContour.ALL in self.disp:
            cont = get_cell_contour(self.markers[self.idx], self.im_cardio, 
                                    self.im_markers, self.im_pxs)
            self.ax_cell.add_patch(cont)
        if SingleCellDisplayContour.COVER in self.disp or SingleCellDisplayContour.ALL in self.disp:
            cont = get_cover_px_contour(self.markers[self.idx], self.im_cardio, 
                                    self.im_markers, self.im_pxs)
            self.ax_cell.add_patch(cont)
        if SingleCellDisplayContour.MAX in self.disp or SingleCellDisplayContour.ALL in self.disp:
            cont = get_max_px_contour(self.markers[self.idx], self.im_cardio, 
                                    self.im_markers, self.im_pxs)
            self.ax_cell.add_patch(cont)
        self.selected_pts.set_data((self.centers[self.selected_coords, 0], self.centers[self.selected_coords, 1]))
        sig = get_max_px_signal_by_cell_id(self.markers[self.idx], self.well, self.im_cardio, 
                                self.im_markers, self.im_pxs)
        self.line_max.set_data((np.linspace(0, self.well.shape[0], self.well.shape[0]), sig))
        self.ax_max.set_ylim((-.05, np.max(sig)))
        sig = get_cover_px_signal_by_cell_id(self.markers[self.idx], self.well, self.im_cardio, 
                                self.im_markers, self.im_pxs)
        self.line_int.set_data((np.linspace(0, self.well.shape[0], self.well.shape[0]), sig))
        self.ax_int.set_ylim((np.min(sig), np.max(sig)))
    #     sig = get_weighted_cover_px_signal_by_cell_id(self.markers[self.idx], self.im_cardio, 
    #                             self.im_markers, self.im_pxs)
    #     elm6.set_data((np.linspace(0, self.well.shape[0], self.well.shape[0]), sig))
    #     ax[2, 1].set_ylim((np.min(sig), np.max(sig)))
        self.fig.canvas.draw()

    def on_button_plus_clicked(self, b):
        self.idx = min(self.idx + 1, len(self.markers) - 1)
        self.draw_plot()
        
    def on_button_minus_clicked(self, b):
        self.idx = max(self.idx - 1, 0)
        self.draw_plot()
        
    def on_button_save_clicked(self, b):
        if self.idx not in self.selected_coords:
            self.selected_coords.append(self.idx)
        self.on_button_plus_clicked(b)

    def on_mouse_button_press(self, evt):
        if evt.inaxes in [self.ax_mic]:
            ret = get_closest_coords((evt.xdata, evt.ydata), self.centers)
            if ret is not None:
                idx, cd = ret
                self.idx = idx
                self.draw_plot()
        
    def on_press(self, event):
        if event.key == 'right' or event.key == '6':
            self.on_button_plus_clicked(None)
        elif event.key == 'left' or event.key == '4':
            self.on_button_minus_clicked(None)
        elif event.key == 'enter':
            self.on_button_save_clicked(None)