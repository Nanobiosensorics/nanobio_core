from __future__ import annotations


class CardioMicScaling:
    MIC_5X = 2134 / 2.22
    MIC_10X = 2134 * 0.9817
    MIC_20X = 2134 * 1.81


def get_scale(scale: float) -> tuple[int, float]:
    mic_px_per_um = float(scale) / 1000.0
    epic_px_per_um = 1 / 25
    epic_cardio_scale = int(80 * (mic_px_per_um / epic_px_per_um))
    mic_um_per_px = 1 / mic_px_per_um
    return epic_cardio_scale, mic_um_per_px


__all__ = ["CardioMicScaling", "get_scale"]
