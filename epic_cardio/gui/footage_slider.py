import numpy as np
from matplotlib.widgets import Slider
from matplotlib.widgets import CheckButtons


class FootageSlider:
    def __init__(
        self,
        fig,
        on_frame_change,
        label='Frame',
        axes_rect=(0.2, 0.08, 0.6, 0.03),
        max_axes_rect=(0.45, 0.02, 0.1, 0.045),
    ):
        self._fig = fig
        self._on_frame_change = on_frame_change
        self._label = label
        self._axes_rect = axes_rect
        self._max_axes_rect = max_axes_rect
        self._slider_ax = None
        self._max_ax = None
        self._slider = None
        self._max_check = None
        self._slider_cid = None
        self._max_cid = None
        self._suspend_callback = False
        self._suspend_max_callback = False
        self._is_max = True

    def remove(self):
        if self._slider is not None and self._slider_cid is not None:
            self._slider.disconnect(self._slider_cid)
        if self._max_check is not None and self._max_cid is not None:
            self._max_check.disconnect(self._max_cid)

        if self._slider_ax is not None:
            try:
                if self._slider_ax.figure is not None:
                    self._slider_ax.remove()
            except Exception:
                pass
        if self._max_ax is not None:
            try:
                if self._max_ax.figure is not None:
                    self._max_ax.remove()
            except Exception:
                pass
        self._slider_ax = None
        self._max_ax = None
        self._slider = None
        self._max_check = None
        self._slider_cid = None
        self._max_cid = None

    def build(self, frame_count, current_frame=0, max_checked=True):
        self.remove()
        self._is_max = bool(max_checked)
        max_frame = max(int(frame_count) - 1, 0)
        frame = int(np.clip(current_frame, 0, max_frame))

        self._slider_ax = self._fig.add_axes(self._axes_rect)
        try:
            self._slider = Slider(
                self._slider_ax,
                self._label,
                valmin=0,
                valmax=max_frame,
                valinit=frame,
                valstep=1,
                valfmt='%d',
                useblit=False,
            )
        except (TypeError, AttributeError):
            self._slider = Slider(
                self._slider_ax,
                self._label,
                valmin=0,
                valmax=max_frame,
                valinit=frame,
                valstep=1,
                valfmt='%d',
            )
        self._slider_cid = self._slider.on_changed(self._on_slider_change)

        self._max_ax = self._fig.add_axes(self._max_axes_rect)
        try:
            self._max_check = CheckButtons(self._max_ax, ['Max'], [self._is_max], useblit=False)
        except (TypeError, AttributeError):
            self._max_check = CheckButtons(self._max_ax, ['Max'], [self._is_max])
        self._max_cid = self._max_check.on_clicked(self._on_max_toggle)
        self._max_ax.set_facecolor('none')
        self._update_slider_style()

    def set_frame(self, frame, emit=False):
        if self._slider is None:
            return
        target = int(np.clip(frame, self._slider.valmin, self._slider.valmax))
        if emit:
            self._slider.set_val(target)
            return

        self._suspend_callback = True
        self._slider.set_val(target)
        self._suspend_callback = False

    def is_max_mode(self):
        return self._is_max

    def set_max_mode(self, is_max, emit=False):
        if self._max_check is None:
            self._is_max = bool(is_max)
            return

        target = bool(is_max)
        if self._is_max != target:
            self._suspend_max_callback = True
            self._max_check.set_active(0)
            self._suspend_max_callback = False
            self._is_max = target

        self._update_slider_style()
        if emit:
            self._emit_change()

    def _on_slider_change(self, value):
        if self._suspend_callback:
            return

        if self._is_max:
            self.set_max_mode(False, emit=True)
            return

        self._emit_change()

    def _on_max_toggle(self, _label):
        if self._suspend_max_callback:
            return
        self._is_max = not self._is_max
        self._update_slider_style()
        self._emit_change()

    def _emit_change(self):
        if self._slider is None:
            return
        self._on_frame_change(int(np.rint(self._slider.val)), self._is_max)

    def _update_slider_style(self):
        if self._slider_ax is None or self._slider is None:
            return

        if self._is_max:
            self._slider_ax.set_facecolor('#e8e8e8')
            if hasattr(self._slider, 'poly') and self._slider.poly is not None:
                self._slider.poly.set_facecolor('#b5b5b5')
                self._slider.poly.set_alpha(0.35)
        else:
            self._slider_ax.set_facecolor('white')
            if hasattr(self._slider, 'poly') and self._slider.poly is not None:
                self._slider.poly.set_facecolor('#6f93b6')
                self._slider.poly.set_alpha(0.85)
        self._fig.canvas.draw_idle()
