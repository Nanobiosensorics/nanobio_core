import numpy as np
from .funcs import *
from ..epic_cardio.math_ops import get_max_well
from datetime import datetime
import os
import pandas as pd
import json
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import cv2

def dice_loss(y, p):
    return np.abs(1 - ((2*np.sum(y*p) + 1) / (np.sum(y) + np.sum(p) + 1)))

class CardioMicSingleCellEvaluator():

    def __init__(self, time, well, im_mic, im_mask, im_pred, params, load_selection=None,
                 save_selection=True, save_path='./segmentation.npz', ws_threshold = 160,
                 filter_params = {
                    'area': (-np.Inf, np.Inf),
                    'max_value': (100, np.Inf),
                    'adjacent': True,
                }):
        self.idx = 0
        self.time = time
        self.well = well
        self.save_path = save_path
        self.selected_coords = []

        if type(params) == dict:
            self.translation = params['translation']
            self.scale = params['scale']
            self.px_size = params['px_size']
        elif os.path.exists(params):
            with open(params, 'r') as fp:
                p = json.load(fp)
                self.translation = np.array([p['t_0'], p['t_1']])
                self.scale = p['scale']
                self.px_size = self.scale / 80 / 25
        else:
            raise ValueError('Wrong value given for params')

        self.save_selection = save_selection

        # Image slicing

        im_pxs = np.asarray([[ n + m * 80 for n in range(0, 80)] for m in range(0, 80) ])
        im_pxs = cv2.resize(im_pxs, (self.scale, self.scale), interpolation=cv2.INTER_NEAREST)
        im_cardio = cv2.resize(get_max_well(well), (self.scale, self.scale), interpolation=cv2.INTER_NEAREST)

        cardio_slice, mic_slice = CardioMicSingleCellEvaluator._get_slicer(im_mask.shape, (self.scale, self.scale), self.translation)

        im_markers = im_mask[mic_slice]
        im_mic = im_mic[mic_slice]
        im_pred = im_pred[mic_slice]
        im_cardio = im_cardio[cardio_slice]
        im_pxs = im_pxs[cardio_slice]

        # Image filtering

        markers, centers = CardioMicSingleCellEvaluator._single_cell_filtering(filter_params, well, im_cardio, im_mask, im_markers, im_pxs, self.translation, (self.scale, self.scale), self.px_size)

        cardio_centers = CardioMicSingleCellEvaluator._get_cardio_centers(centers, im_pxs)
        # print(cardio_centers)
        crd = well[-1].copy()
        crd[crd < 0] = 0
        bn = np.zeros(crd.shape)
        bn[crd > ws_threshold] = 1
        mask = np.zeros(crd.shape, dtype=int)
        for i in range(cardio_centers.shape[1]):
            mask[cardio_centers[0, i], cardio_centers[1, i]] = markers[i]
        im_watershed = watershed(-crd, mask, mask=bn)
        im_watershed *= bn.astype(int)
        # self.cardio_watershed = im_watershed.copy()

        markers, centers = CardioMicSingleCellEvaluator._adjacent_filtering(filter_params, markers, im_markers, im_pxs)

        self.cardio_watershed = np.zeros(im_watershed.shape)
        for marker in markers:
            self.cardio_watershed[im_watershed == marker] = marker

        im_watershed = cv2.resize(self.cardio_watershed, (self.scale, self.scale), interpolation=cv2.INTER_NEAREST)
        im_watershed = im_watershed[cardio_slice]

        self.markers = markers

        im_contour = np.zeros(im_markers.shape).astype('uint8')
        im_contour[im_markers > 0] = 1
        self.im_contour = get_contour(im_contour, 1)

        self.im_cardio, self.im_mic, self.im_markers, self.im_pred, self.im_pxs, self.im_contour, self.centers, self.im_watershed = im_cardio, im_mic, im_markers, im_pred, im_pxs, im_contour, centers, im_watershed

        if load_selection != None:
            if os.path.exists(load_selection):
                with open(load_selection, 'r') as fp:
                    selection = json.load(fp)['ids']
                    for sel in selection:
                        for i, marker in enumerate(markers):
                            if sel == marker:
                                self.selected_coords.append(i)

    @classmethod
    def _get_cardio_centers(cls, centers, im_pxs):
        cntrs= []
        for cnt in centers:
            px_center = im_pxs[int(cnt[1]), int(cnt[0])]
            cntrs.append((px_center % 80, px_center // 80))
        cntrs = np.array(cntrs).T
        cntrs = np.flip(cntrs, axis=0)
        return cntrs

    @classmethod
    def _get_slicer(cls, shape, scale, translation):
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
    def _transform_resolution(cls, im_cardio, im_mic, im_markers, im_pxs, im_contour, im_watershed, centers, shape, resolution = 1):
        if resolution == 1:
            return im_cardio, im_mic, im_markers, im_pxs, im_contour, im_watershed, centers

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
        im_watershed_tr = cv2.resize(im_watershed,
                        (round(shape[1] * resolution), round(shape[0] * resolution)),
                        interpolation=cv2.INTER_NEAREST)

        centers_tr = []
        for coord in centers:
            centers_tr.append(
                (coord[0] / (shape[0]) * (shape[0] * resolution),
                coord[1] / (shape[1]) * (shape[1] * resolution)))
        centers_tr = np.asarray(centers_tr)
        return im_cardio_tr, im_mic_tr, im_markers_tr, im_pxs_tr, im_contour_tr, im_watershed_tr, centers_tr

    @classmethod
    def _single_cell_filtering(cls, filter_params, well, im_cardio_sliced, im_markers, im_markers_sliced, im_pxs_sliced, translation, shape, px_size):
        now = datetime.now()

        markers_filter = im_markers_sliced.copy().astype(int)
        unique_cell = np.unique(markers_filter)
        markers_selected = []
        for n, i in enumerate(unique_cell):
            # print(f'Single cell based filtering: {n + 1}/{len(unique_cell)}', end='\r' if n + 1 != len(unique_cell) else '\n')
            y, x = (im_markers == i).nonzero()
            if np.any(y <= translation[1]) or np.any(y >= translation[1] + shape[1]):
                continue
            if np.any(x <= translation[0]) or np.any(x >= translation[0] + shape[0]):
                continue
            ar = get_area_by_cell_id(i, im_markers, px_size)
            mx = np.max(get_max_px_signal_by_cell_id(i, well, im_cardio_sliced, im_markers_sliced, im_pxs_sliced))
            if (ar > filter_params['area'][0]) & (ar < filter_params['area'][1]) & (mx > filter_params['max_value'][0]) & (mx < filter_params['max_value'][1]):
                markers_selected.append(i)

        # print(f'Duration {datetime.now() - now}')
        # print(f'Cell centers calculation')
        now = datetime.now()
        im_markers_selected = mask_centers(im_markers_sliced, markers_selected)
        # print(f'Duration {datetime.now() - now}')
        return markers_selected, im_markers_selected

    @classmethod
    def _adjacent_filtering(cls, filter_params, markers_selected_sc, im_markers_sliced, im_pxs_sliced):
        now = datetime.now()

        markers_filter = im_markers_sliced.copy().astype(int)

        if filter_params['adjacent']:
            markers_selected = []
            for n, i in enumerate(markers_selected_sc):
                # print(f'Relational filtering: {n + 1}/{len(markers_selected_sc)}', end='\r' if n + 1 != len(markers_selected_sc) else '\n')
                ngh = is_adjacent(i, markers_filter, im_pxs_sliced)
                ngh = set(ngh) - set([i])
                if len(ngh) != 0:
                    # print(i)
                    continue
                markers_selected.append(i)
        else:
            markers_selected = markers_selected_sc
        # print(f'Duration {datetime.now() - now}')
        # print(f'Cell centers calculation')
        now = datetime.now()
        im_markers_selected = mask_centers(im_markers_sliced, markers_selected)
        # print(f'Duration {datetime.now() - now}')
        return markers_selected, im_markers_selected

    def select_all(self):
        self.selected_coords = list(range(len(self.markers)))

    def save(self, path, well_id = 'well', px_range = 3):
        if len(self.selected_coords) > 0:
            slaced_px_range = int(self.scale / 80 * px_range)
            max_signals = np.zeros((len(self.selected_coords), self.well.shape[0]), dtype=np.float32)
            cover_signals = np.zeros((len(self.selected_coords), self.well.shape[0]), dtype=np.float32)
            # weigthed_cover_signals = np.zeros((len(selected_coords), im_src.shape[0]))
            cell_areas = np.zeros(len(self.selected_coords), dtype=np.float32)
            cell_mic_centers = np.zeros((len(self.selected_coords), 2), dtype=np.uint16)
            cell_cardio_centers = np.zeros((len(self.selected_coords), 2), dtype=np.uint8)
            cell_mics = np.zeros((len(self.selected_coords), 2*slaced_px_range, 2*slaced_px_range, 3), dtype=np.uint8)
            cell_mics_singular = np.zeros((len(self.selected_coords), 2*slaced_px_range, 2*slaced_px_range, 3), dtype=np.uint8)
            cell_markers = np.zeros((len(self.selected_coords), 2*slaced_px_range, 2*slaced_px_range), dtype=np.uint16)
            cell_markers_singular = np.zeros((len(self.selected_coords), 2*slaced_px_range, 2*slaced_px_range), dtype=np.uint16)
            cell_cardio = np.zeros((len(self.selected_coords), self.well.shape[0], 2*px_range, 2*px_range), dtype=np.float32)
            cell_cover = np.zeros((len(self.selected_coords), self.well.shape[0], 2*px_range, 2*px_range), dtype=np.float32)
            cell_pred = []
            cell_watershed = []# np.zeros((len(self.selected_coords), self.well.shape[0], 2*px_range, 2*px_range), dtype=np.float32)
            # now = datetime.now()

            selection = []

            dl_p = []
            dl_cp = []
            dl_w = []

            for i, selected_id in enumerate(sorted(self.selected_coords)):
                # print(f'Progress {i + 1}/{len(self.selected_coords)}', end='\r')
                cell_id = self.markers[selected_id]
                cell_center = self.centers[selected_id].astype(int)
                max_signals[i, :] = get_max_px_signal_by_cell_id(cell_id, self.well, self.im_cardio, self.im_markers, self.im_pxs)
                cover_signals[i, :] = get_cover_px_signal_by_cell_id(cell_id, self.well, self.im_cardio, self.im_markers, self.im_pxs)
            #     weigthed_cover_signals[i, :] = get_weighted_cover_px_signal_by_cell_id(self.markers[cell_id], self.im_cardio, self.im_markers, self.im_pxs)
                cell_areas[i] = get_area_by_cell_id(cell_id, self.im_markers, self.px_size)
                px_center = self.im_pxs[cell_center[1], cell_center[0]]
                cardio_center = (px_center % 80, px_center // 80)
                cell_cardio_centers[i] = cardio_center
                cell_mic_centers[i] = cell_center

                ranges = ((max(cell_center[1] - slaced_px_range, 0), min(cell_center[1] + slaced_px_range, self.im_mic.shape[0])),
                          (max(cell_center[0] - slaced_px_range, 0), min(cell_center[0] + slaced_px_range, self.im_mic.shape[1])))
                mic_slice = (slice(*ranges[0]), slice(*ranges[1]))
                mic_slice_proj = (slice(0, ranges[0][1] - ranges[0][0]), slice(0, ranges[1][1] - ranges[1][0]))

                ranges = ((max(cardio_center[1] - px_range, 0), min(cardio_center[1] + px_range, self.well.shape[1])),
                          (max(cardio_center[0] - px_range, 0), min(cardio_center[0] + px_range, self.well.shape[2])))
                cardio_slice = (slice(*ranges[0]), slice(*ranges[1]))
                cardio_slice_proj = (slice(0, ranges[0][1] - ranges[0][0]), slice(0, ranges[1][1] - ranges[1][0]))

                cell_mics[i, mic_slice_proj[0], mic_slice_proj[1]] = self.im_mic[mic_slice[0], mic_slice[1]]
                cell_markers[i, mic_slice_proj[0], mic_slice_proj[1]] = self.im_markers[mic_slice[0], mic_slice[1]]
                cell_mics_singular[i] = cell_mics[i].copy()
                cell_mics_singular[i][cell_markers[i] != cell_id] = 0
                cell_markers_singular[i] = cell_markers[i] == cell_id

                cardio_im = np.zeros((self.well.shape[0], 2*px_range, 2*px_range), dtype=np.float32)
                cardio_im[:, cardio_slice_proj[0], cardio_slice_proj[1]] = self.well[:, cardio_slice[0], cardio_slice[1]]
                cell_cardio[i] = cardio_im

                cover_im = np.zeros((self.well.shape[0], 2*px_range, 2*px_range), dtype=np.float32)
                cover_im[:, cardio_slice_proj[0], cardio_slice_proj[1]] = get_cover_px_well_by_cell_id(cell_id, self.well, self.im_markers, self.im_pxs)[:, cardio_slice[0], cardio_slice[1]]
                cell_cover[i] = cover_im

                pred_im = np.zeros((self.well.shape[0], 2*px_range, 2*px_range), dtype=np.float32)
                pred_im[:, cardio_slice_proj[0], cardio_slice_proj[1]] = get_cover_px_well_by_cell_id(cell_id, self.well, self.im_pred, self.im_pxs)[:, cardio_slice[0], cardio_slice[1]]
                if np.sum(pred_im) != 0:
                    dl_p.append(dice_loss(self.im_markers[mic_slice[0], mic_slice[1]] > 0, self.im_pred[mic_slice[0], mic_slice[1]] > 0))
                    dl_cp.append(dice_loss(cover_im > 0, pred_im > 0))
                    cell_pred.append(pred_im)

                ws_im = np.zeros((self.well.shape[0], 2*px_range, 2*px_range), dtype=np.float32)
                well_ws = self.well.copy()
                well_ws[:, self.cardio_watershed != cell_id] = 0
                ws_im[:, cardio_slice_proj[0], cardio_slice_proj[1]] = well_ws[:, cardio_slice[0], cardio_slice[1]]
                if np.sum(ws_im) != 0:
                    dl_w.append(dice_loss(ws_im > 0, cardio_im > 0))
                    cell_watershed.append(ws_im)

                selection.append(int(cell_id))

            de_p = 1 - (len(cell_pred) / len(selection))
            de_w = 1 - (len(cell_watershed) / len(selection))
            dl_p_s = [np.mean(dl_p), np.std(dl_p)]
            dl_cp_s = [np.mean(dl_cp), np.std(dl_cp)]
            dl_w_s = [np.mean(dl_w), np.std(dl_w)]

            cell_pred = np.asarray(cell_pred)
            cell_watershed = np.asarray(cell_watershed)
            # print(de_p, de_w, dl_p_s, dl_cp_s, dl_w_s)
            # cell mic image, cardio video, coordinates
            # print(f'Duration {datetime.now() - now}')
            pd.DataFrame(np.vstack((self.time, max_signals))).to_csv(os.path.join(path, f'{well_id}_max_signals.csv'))
            pd.DataFrame(np.vstack((self.time, cover_signals))).to_csv(os.path.join(path, f'{well_id}_int_signals.csv'))
            pd.DataFrame(cell_areas).to_csv(os.path.join(path, f'{well_id}_areas.csv'))
            pd.DataFrame(cell_mic_centers).to_csv(os.path.join(path, f'{well_id}_mic_centers.csv'))
            pd.DataFrame(cell_cardio_centers).to_csv(os.path.join(path, f'{well_id}_cardio_centers.csv'))
            print(cell_mics.shape)
            np.savez(os.path.join(path, f'{well_id}_seg.npz'), time=self.time, cardio=cell_cardio, cardio_watershed=cell_watershed, cardio_cover=cell_cover, cardio_pred=cell_pred,
                     mic=cell_mics, mic_singular=cell_mics_singular, marker=cell_markers, marker_singular=cell_markers_singular, well=self.well, im_markers=self.im_markers, im_mic=self.im_mic)

            # print(cell_cardio.shape, cell_cover.shape, cell_pred.shape, cell_watershed.shape)
            # np.savez(os.path.join(path, f'{well_id}_seg.npz'), time=self.time, cardio=cell_cardio, cardio_watershed=cell_watershed, cardio_cover=cell_cover, cardio_pred=cell_pred)

            sel_path = os.path.join(path, 'metadata')
            if not os.path.exists(sel_path):
                os.makedirs(sel_path)

            with open(os.path.join(sel_path, f'{well_id}_selection.json'), 'w') as fp:
                json.dump({'ids': selection}, fp)
            with open(os.path.join(sel_path, f'{well_id}_stats.json'), 'w') as fp:
                json.dump({'de_p': de_p,
                           'de_w': de_w,
                           'dl_p': dl_p_s,
                           'dl_cp': dl_cp_s,
                           'dl_w': dl_w_s}, fp)


def evaluate(*args, **kwargs):
    """Stable functional entry point for single-cell evaluation."""
    return CardioMicSingleCellEvaluator(*args, **kwargs)
