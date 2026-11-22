"""The original V2 30->25 Hz motion preprocessing, with recorded diagnostics."""
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

PREPROCESS = dict(version=1, source_fps=30, target_fps=25, interpolation="cubic",
                  face_hz=7, head_hz=3, eye_hz=2, order=4, roll_median=5,
                  blendshape_clip=[0, 1], dtype="float32")


def prepare_motion(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 61 or len(raw) < 4:
        raise ValueError("Raw motion must have shape [T>=4,61]")
    n = int(np.floor((len(raw) - 1) / 30 * 25)) + 1
    old, new = np.arange(len(raw)) / 30., np.arange(n) / 25.
    result = np.empty((n, 61), dtype=np.float64)
    for c in range(61):
        col = raw[:, c].copy()
        mask = np.isnan(col)
        if mask.all():
            result[:, c] = np.nan
            continue
        if mask.any():
            col[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), col[~mask])
        result[:, c] = interp1d(old, col, kind="cubic", assume_sorted=True,
                               bounds_error=False, fill_value="extrapolate")(new)
    nonfinite = int((~np.isfinite(result)).sum())
    for c in range(61):
        valid = np.isfinite(result[:, c])
        if not valid.all():
            result[:, c] = np.interp(np.arange(n), np.flatnonzero(valid), result[valid, c]) if valid.any() else 0
    if n <= 15:
        raise ValueError("Motion is too short for the original zero-phase filter")
    for c in (57, 60):
        result[:, c] = median_filter(result[:, c], size=5, mode="nearest")
    for group, hz in ((slice(0, 52), 7), (slice(52, 55), 3), (slice(55, 61), 2)):
        result[:, group] = sosfiltfilt(butter(4, hz, fs=25, btype="low", output="sos"), result[:, group], axis=0)
    result[:, :52] = np.clip(result[:, :52], 0, 1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite prepared motion")
    return result.astype(np.float32), {"raw_nonfinite": int((~np.isfinite(raw)).sum()), "filled_after_resampling": nonfinite}
