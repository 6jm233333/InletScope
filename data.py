from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class WindowedRun:

    def __init__(self, run_id, time, pressure, state, task, window=100,
                 stride=20, horizon=.5, sample_rate=2000, sensor_ids=None):
        if task not in {"detection", "warning"}:
            raise ValueError("task must be 'detection' or 'warning'")
        if not isinstance(window, (int, np.integer)) or window < 1:
            raise ValueError("window must be a positive integer")
        if not isinstance(stride, (int, np.integer)) or stride < 1:
            raise ValueError("stride must be a positive integer")
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("sample_rate must be positive and finite")
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("horizon must be positive and finite")
        t = np.asarray(time, dtype=np.float64)
        x = np.asarray(pressure)
        raw_state = np.asarray(state)
        if t.ndim != 1 or len(t) < window:
            raise ValueError("time must be one-dimensional with >= window samples")
        if x.ndim != 2 or x.shape[0] != len(t) or x.shape[1] < 1:
            raise ValueError("pressure must have shape [sample, nonempty sensor]")
        if raw_state.ndim != 1 or len(raw_state) != len(t):
            raise ValueError("state must be one-dimensional and aligned with time")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)):
            raise ValueError("time and pressure must be finite")
        if not np.all(np.isin(raw_state, [0, 1])):
            raise ValueError("state values must be binary: normal=0, unstart=1")
        dt = np.diff(t)
        if np.any(dt <= 0) or not np.allclose(
                dt, 1.0 / sample_rate, rtol=1e-5, atol=1e-10):
            raise ValueError("time must be increasing and uniformly sampled at sample_rate")
        if any(np.allclose(x[:, j], t, rtol=1e-7, atol=1e-8)
               for j in range(x.shape[1])):
            raise ValueError("pressure contains an apparent time column")
        if sensor_ids is None:
            sensor_ids = [f"S{i + 1:02d}" for i in range(x.shape[1])]
        sensor_ids = list(sensor_ids)
        if (len(sensor_ids) != x.shape[1]
                or len(set(sensor_ids)) != len(sensor_ids)
                or any(not isinstance(s, str) or not s for s in sensor_ids)):
            raise ValueError("sensor_ids must be unique nonempty names aligned with pressure")

        self.run = str(run_id)
        self.raw_time = t
        self.raw_state = raw_state.astype(np.int8, copy=False)
        self.values = x.astype(np.float32, copy=False)
        self.sensor_ids = sensor_ids
        self.task = task
        self.sample_rate = sample_rate
        self.window = window
        self.stride = stride
        self.horizon = horizon
        onset_indices = np.flatnonzero(np.diff(self.raw_state.astype(int)) == 1) + 1
        self.onsets = t[onset_indices]
        starts = np.arange(0, len(t) - window + 1, stride)
        ends = starts + window - 1
        self.time = t[ends]
        self.windows = np.lib.stride_tricks.sliding_window_view(
            self.values, window, axis=0)[starts]
        prefix = np.concatenate(([0], np.cumsum(self.raw_state, dtype=np.int64)))
        observed_positive = (prefix[ends + 1] - prefix[starts]) > 0
        if task == "detection":
            self.y = observed_positive.astype(np.int64)
            self.eligible = np.ones(len(starts), dtype=bool)
        else:
            horizon_samples = horizon * sample_rate
            rounded_horizon = round(horizon_samples)
            if not np.isclose(horizon_samples, rounded_horizon, atol=1e-8, rtol=0):
                raise ValueError("horizon must be an integer number of samples")
            future_ends = ends + rounded_horizon
            self.eligible = (~observed_positive) & (future_ends < len(t))
            left = np.searchsorted(onset_indices, ends, side="right")
            right = np.searchsorted(onset_indices, future_ends, side="right")
            self.y = (right > left).astype(np.int64)


def load_config(path):

    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    for field in ("dataset", "data_kind", "settings", "split", "runs", "training", "seeds"):
        if field not in config:
            raise ValueError(f"missing configuration field: {field}")
    run_groups = []
    for split in ("train", "validation", "test"):
        ids = config["split"].get(split)
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
            raise ValueError(f"{split} must list unique run IDs")
        run_groups.extend(ids)
    if len(set(run_groups)) != len(run_groups):
        raise ValueError("train, validation and test runs must be disjoint")
    if set(run_groups) != set(config["runs"]):
        raise ValueError("each configured run must belong to exactly one split")
    for run_id, run in config["runs"].items():
        sensors = run.get("sensor_ids", [])
        if (not sensors or len(set(sensors)) != len(sensors)
                or any(not isinstance(s, str) or not s for s in sensors)):
            raise ValueError(f"invalid sensor_ids for {run_id}")
        configurations = run.get("configurations", {})
        if "all" not in configurations or set(configurations["all"]) != set(sensors):
            raise ValueError(f"{run_id}: 'all' must contain all sensor IDs")
        for name, subset in configurations.items():
            if (not isinstance(subset, list) or not subset
                    or len(set(subset)) != len(subset)
                    or not set(subset).issubset(sensors)):
                raise ValueError(f"invalid sensor configuration: {run_id}/{name}")
    return config


def load_run(dataset_path, run_id, task, settings, sensor_ids=None):
    
    with np.load(dataset_path, allow_pickle=False) as data:
        prefix = f"{run_id}__"
        required = [prefix + name for name in ("time", "pressure", "state")]
        if any(key not in data for key in required):
            raise ValueError(f"dataset is missing arrays for run {run_id}")
        arrays = [data[key] for key in required]
    allowed = {key: settings[key] for key in
               ("window", "stride", "horizon", "sample_rate") if key in settings}
    return WindowedRun(run_id, *arrays, task=task, sensor_ids=sensor_ids, **allowed)
