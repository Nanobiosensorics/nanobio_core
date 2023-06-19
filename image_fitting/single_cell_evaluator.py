import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import numpy as np
from .funcs import *
from ..epic_cardio.math_ops import get_max_well
from datetime import datetime
import os
import pandas as pd

class SingleCellDisplayContour:
    CELL = 0,
    MAX = 1,
    COVER = 2,
    ALL = 255

class CardioMicSingleCellEvaluator():
    
    def __init__(self, well, im_mic, im_mask, scale, translation, px_size, 
                 display_contours: list = [
                    SingleCellDisplayContour.ALL,   
                 ]):
        self.disp = display_contours
        self.idx = 0
        self.well = well
        self.translation = translation
        self.scale = scale
        
        # Image slicing

        im_pxs = np.asarray([[ n + m * 80 for n in range(0, 80)] for m in range(0, 80) ])
        im_pxs = cv2.resize(im_pxs, (scale, scale), interpolation=cv2.INTER_NEAREST)
        im_cardio = cv2.resize(get_max_well(well), (scale, scale), interpolation=cv2.INTER_NEAREST)
        
        cardio_slice, mic_slice = CardioMicSingleCellEvaluator._get_slicer(im_mask.shape, (scale, scale), translation)

        im_markers = im_mask[mic_slice]
        im_mic = im_mic[mic_slice]
        im_cardio = im_cardio[cardio_slice]
        im_pxs = im_pxs[cardio_slice]
        
        # Image filtering
        
        filter_params = {
            'area': (-np.Inf, np.Inf),
            'max_value': (50, np.Inf),
            'adjacent': False,
        }
        
        markers, centers = CardioMicSingleCellEvaluator._cell_filtering(filter_params, well, im_cardio, im_mask, im_markers, im_pxs, translation, (scale, scale), px_size)
        
        self.markers = markers
        
        
        im_contour = np.zeros(im_markers.shape).astype('uint8')
        im_contour[im_markers > 0] = 1
        self.im_contour = get_contour(im_contour, 1)
        
        self.im_cardio, self.im_mic, self.im_markers, self.im_pxs, self.im_contour, self.centers = im_cardio, im_mic, im_markers, im_pxs, im_contour, centers
        
        self.px_size = px_size
        self.selected_coords = []
    
    @classmethod
    def _get_slicer(self, shape, scale, translation):
        start_mic = np.array((max(translation[1], 0), 
                    max(translation[0], 0)))
        end_mic = np.flipud((translation + scale))
        start_cardio = np.abs((min(0, translation[1]), min(0, translation[0])))
        over_reach = (-(shape - np.flipud(scale + translation))).clip(min=0)
        end_cardio = np.flipud(scale) - over_reach
        mic_slice = (slice(start_mic[0], end_mic[0]), slice(start_mic[1], end_mic[1]))
        cardio_slice = (slice(start_cardio[0], end_cardio[0]), slice(start_cardio[1], end_cardio[1]))
        return cardio_slice, mic_slice
        
    @classmethod 
    def _transform_resolution(self, im_cardio, im_mic, im_markers, im_pxs, im_contour, centers, shape, resolution = 1):
        if resolution == 1:
            return im_cardio, im_mic, im_markers, im_pxs, im_contour, centers

        im_contour = np.zeros(im_markers.shape).astype('uint8')
        im_contour[im_markers > 0] = 1
        im_contour = get_contour(im_contour, 1)
        

        im_cardio_tr = cv2.resize(im_cardio, 
                        (round(shape[1] * resolution), round(shape[0] * resolution)), 
                        interpolation=cv2.INTER_NEAREST)
        im_mic_tr = cv2.resize(im_mic, 
                        (round(shape[1] * resolution), round(shape[0] * resolution)), 
                        interpolation=cv2.INTER_NEAREST)
        im_markers_tr = cv2.resize(im_markers.astype('uint16'), 
                            (round(shape[1] * resolution), round(shape[0] * resolution)), 
                            interpolation=cv2.INTER_NEAREST)
        im_pxs_tr = cv2.resize(im_pxs.astype('uint16'), 
                            (round(im_pxs.shape[1] * resolution), round(im_pxs.shape[0] * resolution)), 
                            interpolation=cv2.INTER_NEAREST)
        im_contour_tr = cv2.resize(im_contour, 
                            (round(shape[1] * resolution), round(shape[0] * resolution)), 
                            interpolation=cv2.INTER_NEAREST)

        centers_tr = []
        for coord in centers:
            centers_tr.append(
                (coord[0] / (shape[0]) * (shape[0] * resolution), 
                coord[1] / (shape[1]) * (shape[1] * resolution)))
        centers_tr = np.asarray(centers_tr)
        return im_cardio_tr, im_mic_tr, im_markers_tr, im_pxs_tr, im_contour_tr, centers_tr
    
    @classmethod
    def _cell_filtering(self, filter_params, well, im_cardio_sliced, im_markers, im_markers_sliced, im_pxs_sliced, translation, shape, px_size):
        now = datetime.now()

        markers_filter = im_markers_sliced.copy().astype(int)
        unique_cell = np.unique(markers_filter)
        markers_selected_1 = []
        markers_excluded = []
        for n, i in enumerate(unique_cell):
            print(f'Single cell based filtering: {n + 1}/{len(unique_cell)}', end='\r' if n + 1 != len(unique_cell) else '\n')
            y, x = (im_markers == i).nonzero()
            if np.any(y <= translation[1]) or np.any(y >= translation[1] + shape[1]):
                markers_excluded.append(i)
                continue
            if np.any(x <= translation[0]) or np.any(x >= translation[0] + shape[0]):
                markers_excluded.append(i)
                continue
            ar = get_area_by_cell_id(i, im_markers, px_size)
            mx = np.max(get_max_px_signal_by_cell_id(i, well, im_cardio_sliced, im_markers_sliced, im_pxs_sliced))
            if (ar > filter_params['area'][0]) & (ar < filter_params['area'][1]) & (mx > filter_params['max_value'][0]) & (mx < filter_params['max_value'][1]):
                markers_selected_1.append(i)
                continue
            else:
                markers_excluded.append(i)
                
        if filter_params['adjacent']:
            markers_selected = []
            for n, i in enumerate(markers_selected_1):
                print(f'Relational filtering: {n + 1}/{len(markers_selected_1)}', end='\r' if n + 1 != len(markers_selected_1) else '\n')
                ngh = is_adjacent(i, markers_filter, im_pxs_sliced)
                ngh = set(ngh) - set(markers_excluded) - set([i])
                if len(ngh) != 0:
                    continue
                markers_selected.append(i)
        else:
            markers_selected = markers_selected_1
        print(f'Duration {datetime.now() - now}')
        print(f'Cell centers calculation')
        now = datetime.now()
        im_markers_selected = mask_centers(im_markers_sliced, markers_selected)
        print(f'Duration {datetime.now() - now}')
        return markers_selected, im_markers_selected
    
    def display(self, resolution = 1, px_range = 5):
        
        # Image transformation
        
        self.resolution = resolution
        self.px_range = px_range
        self.scaled_px_range = int(self.scale / 80 * px_range)
        
        self.im_cardio_tr, self.im_mic_tr, self.im_markers_tr, \
            self.im_pxs_tr, self.im_contour_tr, self.centers_tr = CardioMicSingleCellEvaluator._transform_resolution(self.im_cardio, \
                self.im_mic, self.im_markers, self.im_pxs, self.im_contour, self.centers, self.im_cardio.shape, resolution)
            
        # Display
        
        self.fig, self.ax = plt.subplots(2, 2, figsize=(16, 12))
        self.ax_mic = self.ax[0, 0]
        self.ax_cell = self.ax[1, 0]
        self.ax_max = self.ax[0, 1]
        self.ax_int = self.ax[1, 1]
        self.ax_mic.axes.get_xaxis().set_visible(False)
        self.ax_mic.axes.get_yaxis().set_visible(False)
        self.ax_cell.axes.get_xaxis().set_visible(False)
        self.ax_cell.axes.get_yaxis().set_visible(False)
        self.ax_mic.set_position([0, 0.525, .45, .45])
        self.ax_cell.set_position([0, 0.025, .45, .45])

        self.ax_mic.imshow(self.im_mic_tr)
        self.ax_mic.imshow(self.im_cardio_tr, alpha=.4, 
                        vmin = np.mean(self.im_cardio_tr) - 3*np.std(self.im_cardio_tr), 
                        vmax = np.mean(self.im_cardio_tr) + 3*np.std(self.im_cardio_tr))
        self.ax_cell.imshow(self.im_mic_tr)
        self.ax_cell.imshow(self.im_cardio_tr, alpha=.4, 
                        vmin = np.mean(self.im_cardio_tr) - 3*np.std(self.im_cardio_tr), 
                        vmax = np.mean(self.im_cardio_tr) + 3*np.std(self.im_cardio_tr))
        self.ax_max.set_ylim((-.05, 2500))
        self.ax_int.set_ylim((np.min(self.well), np.max(self.well)))
        self.ax_mic.set_title('Cells')
        self.ax_max.set_title('Max pixel signal')
        self.ax_int.set_title('Integrated pixels signal')
        # ax[2, 1].set_title('Integrated weighted pixels signal')
        self.disp_pts, = self.ax_mic.plot(self.centers_tr[:, 0], self.centers_tr[:, 1], 'bo', markersize=3)
        self.crnt_pt, = self.ax_mic.plot(self.centers_tr[0, 0],self.centers_tr[0, 1], 'bo', markersize=10, mfc='none')
        self.selected_pts, =  self.ax_mic.plot( [], [], 'ro', markersize=3)
        self.line_max, = self.ax_max.plot(np.linspace(0, self.well.shape[0], self.well.shape[0]), self.well[:, 0, 0])
        self.line_int, = self.ax_int.plot(np.linspace(0, self.well.shape[0], self.well.shape[0]), self.well[:, 0, 0])
        # elm6, = ax[2, 1].plot(np.linspace(0, self.well.shape[0], self.well.shape[0]), self.well[:, 0, 0])
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_button_press)

        self.draw_plot()
        
    def change_ax_limit(self, x_center, y_center):
        self.ax_cell.set_xlim((x_center - self.scaled_px_range * self.resolution, x_center + self.scaled_px_range * self.resolution))
        self.ax_cell.set_ylim((y_center + self.scaled_px_range * self.resolution, y_center - self.scaled_px_range * self.resolution))
        self.crnt_pt.set_data(((x_center),(y_center)))

    def draw_plot(self):
        self.ax_cell.patches = []
        self.ax_cell.set_title(f'Cell {self.idx + 1}/{len(self.markers)}, Area {np.round(get_area_by_cell_id(self.markers[self.idx], self.im_markers_tr, self.px_size), 2)} μm²')
        self.change_ax_limit(self.centers_tr[self.idx, 0], self.centers_tr[self.idx, 1])
        if SingleCellDisplayContour.CELL in self.disp or SingleCellDisplayContour.ALL in self.disp:
            cont = get_cell_contour(self.markers[self.idx], self.im_cardio_tr, 
                                    self.im_markers_tr, self.im_pxs_tr)
            self.ax_cell.add_patch(cont)
        if SingleCellDisplayContour.COVER in self.disp or SingleCellDisplayContour.ALL in self.disp:
            cont = get_cover_px_contour(self.markers[self.idx], self.im_cardio_tr, 
                                    self.im_markers_tr, self.im_pxs_tr)
            self.ax_cell.add_patch(cont)
        if SingleCellDisplayContour.MAX in self.disp or SingleCellDisplayContour.ALL in self.disp:
            cont = get_max_px_contour(self.markers[self.idx], self.im_cardio_tr, 
                                    self.im_markers_tr, self.im_pxs_tr)
            self.ax_cell.add_patch(cont)
        self.selected_pts.set_data((self.centers_tr[self.selected_coords, 0], self.centers_tr[self.selected_coords, 1]))
        sig = get_max_px_signal_by_cell_id(self.markers[self.idx], self.well, self.im_cardio_tr, 
                                self.im_markers_tr, self.im_pxs_tr)
        self.line_max.set_data((np.linspace(0, self.well.shape[0], self.well.shape[0]), sig))
        self.ax_max.set_ylim((-.05, np.max(sig)))
        sig = get_cover_px_signal_by_cell_id(self.markers[self.idx], self.well, self.im_cardio_tr, 
                                self.im_markers_tr, self.im_pxs_tr)
        self.line_int.set_data((np.linspace(0, self.well.shape[0], self.well.shape[0]), sig))
        self.ax_int.set_ylim((np.min(sig), np.max(sig)))
    #     sig = get_weighted_cover_px_signal_by_cell_id(self.markers[self.idx], self.im_cardio_tr, 
    #                             self.im_markers_tr, self.im_pxs_tr)
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
            ret = get_closest_coords((evt.xdata, evt.ydata), self.centers_tr)
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
            
    def save(self, path, px_range = 3):
        if len(self.selected_coords) > 0:
            slaced_px_range = int(self.scale / 80 * px_range)
            max_signals = np.zeros((len(self.selected_coords), self.well.shape[0]))
            cover_signals = np.zeros((len(self.selected_coords), self.well.shape[0]))
            # weigthed_cover_signals = np.zeros((len(selected_coords), im_src.shape[0]))
            cell_areas = np.zeros(len(self.selected_coords))
            cell_mic_centers = np.zeros((len(self.selected_coords), 2))
            cell_cardio_centers = np.zeros((len(self.selected_coords), 2))
            cell_mics = np.zeros((len(self.selected_coords), 2*slaced_px_range, 2*slaced_px_range, 3))
            cell_mics_singular = np.zeros((len(self.selected_coords), 2*slaced_px_range, 2*slaced_px_range, 3))
            cell_markers = np.zeros((len(self.selected_coords), 2*slaced_px_range, 2*slaced_px_range))
            cell_markers_singular = np.zeros((len(self.selected_coords), 2*slaced_px_range, 2*slaced_px_range))
            cell_cardio = np.zeros((len(self.selected_coords), self.well.shape[0], 2*px_range, 2*px_range))
            now = datetime.now()
            for i, selected_id in enumerate(sorted(self.selected_coords)):
                print(f'Progress {i + 1}/{len(self.selected_coords)}', end='\r')
                cell_id = self.markers[selected_id]
                cell_center = self.centers[selected_id]
                max_signals[i, :] = get_max_px_signal_by_cell_id(cell_id, self.well, self.im_cardio, self.im_markers, self.im_pxs)
                cover_signals[i, :] = get_cover_px_signal_by_cell_id(cell_id, self.well, self.im_cardio, self.im_markers, self.im_pxs)
            #     weigthed_cover_signals[i, :] = get_weighted_cover_px_signal_by_cell_id(self.markers[cell_id], self.im_cardio, self.im_markers, self.im_pxs)
                cell_areas[i] = get_area_by_cell_id(cell_id, self.im_markers, self.px_size)
                cell_mic_centers[i] = cell_center
                cardio_center = (cell_center / self.scale * 80).astype(int)
                cell_center = (cardio_center / 80 * self.scale).astype(int)
                cell_cardio_centers[i] = cardio_center
                
                cell_mics[i] = self.im_mic[cell_center[1] - slaced_px_range : cell_center[1] + slaced_px_range, cell_center[0] - slaced_px_range: cell_center[0] + slaced_px_range]
                cell_markers[i] = self.im_markers[cell_center[1] - slaced_px_range : cell_center[1] + slaced_px_range, cell_center[0] - slaced_px_range: cell_center[0] + slaced_px_range]
                cell_mics_singular[i] = cell_mics[i].copy()
                cell_mics_singular[i][cell_markers[i] != cell_id] = 0
                cell_markers_singular[i] = cell_markers[i] == cell_id
                cell_cardio[i] = self.well[:, cardio_center[1] - px_range : cardio_center[1] + px_range, cardio_center[0] - px_range: cardio_center[0] + px_range]
                
                # cell mic image, cardio video, coordinates
            print(f'Duration {datetime.now() - now}')
            pd.DataFrame(max_signals).to_csv(os.path.join(path, 'max_signals.csv'))
            pd.DataFrame(cover_signals).to_csv(os.path.join(path, 'int_signals.csv'))
            pd.DataFrame(cell_areas).to_csv(os.path.join(path, 'areas.csv'))
            pd.DataFrame(cell_mic_centers).to_csv(os.path.join(path, 'mic_centers.csv'))
            pd.DataFrame(cell_cardio_centers).to_csv(os.path.join(path, 'cardio_centers.csv'))
            np.savez(os.path.join(path, 'segmentation.npz'), cardio=cell_cardio, mic=cell_mics, mic_singular=cell_mics_singular, marker=cell_markers, marker_singular=cell_markers_singular)