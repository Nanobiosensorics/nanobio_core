import numpy as np


class WellPreviewSceneMixin:
    def _init_preview_state(self):
        self._preview_mode = False
        self._preview_axes_by_well = {}
        self._preview_well_by_axes = {}

    def _set_active_well_by_name(self, well_name):
        if well_name in self._ids:
            self._well_id = self._ids.index(well_name)

    def _preview_is_selectable(self, well_name):
        return True

    def _preview_well_image(self, well_name):
        raise NotImplementedError

    def _preview_well_title(self, well_name):
        return well_name

    def _draw_preview_overlay(self, ax, well_name):
        return

    def _build_main_layout(self):
        raise NotImplementedError

    def _redraw_main_scene_on_preview_exit(self):
        raise NotImplementedError

    def _draw_preview_scene(self):
        self._fig.clf()
        self._preview_axes_by_well = {}
        self._preview_well_by_axes = {}

        for n, well_name in enumerate(self._ids):
            ax = self._fig.add_subplot(3, 4, n + 1)
            well_mx = self._preview_well_image(well_name)
            vmax = np.max(well_mx)
            ax.imshow(well_mx, vmin=0, vmax=vmax if vmax > 0 else 1)
            ax.set_title(self._preview_well_title(well_name), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

            self._draw_preview_overlay(ax, well_name)

            is_active = (well_name == self._ids[self._well_id])
            for spine in ax.spines.values():
                spine.set_color('red' if is_active else 'black')
                spine.set_linewidth(2 if is_active else 1)

            self._preview_axes_by_well[well_name] = ax
            self._preview_well_by_axes[ax] = well_name

        self._fig.canvas.draw()

    def _enter_preview_mode(self):
        self._preview_mode = True
        self._draw_preview_scene()

    def _exit_preview_mode(self):
        self._preview_mode = False
        self._build_main_layout()
        self._redraw_main_scene_on_preview_exit()

    def _toggle_preview_mode(self):
        if self._preview_mode:
            self._exit_preview_mode()
        else:
            self._enter_preview_mode()

    def _handle_preview_click(self, event):
        if not self._preview_mode:
            return False
        if hasattr(event, 'button') and event.button != 1:
            return True
        well_name = self._preview_well_by_axes.get(event.inaxes)
        if well_name is None:
            return True
        if self._preview_is_selectable(well_name):
            self._set_active_well_by_name(well_name)
            if getattr(event, 'dblclick', False):
                self._exit_preview_mode()
            else:
                self._draw_preview_scene()
        return True
