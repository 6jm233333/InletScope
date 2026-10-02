from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def configurations(sensor_ids, seed):

    rng = np.random.default_rng(seed)
    result = {"all": list(sensor_ids)}
    for count in (1, 2, 4, 8):
        if count >= len(sensor_ids):
            continue
        combinations = list(itertools.combinations(sensor_ids, count))
        chosen = rng.choice(len(combinations), size=min(20, len(combinations)), replace=False)
        for index, choice in enumerate(sorted(chosen)):
            result[f"count_{count}_{index:02d}"] = list(combinations[choice])
    return result


def generate(output_dir, force=False):
    output_dir = Path(output_dir).resolve()
    config_path = output_dir / "config.json"
    dataset_path = output_dir / "data" / "synthetic.npz"
    existing = [str(p) for p in (config_path, dataset_path) if p.exists()]
    if existing and not force:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing)
                              + ". Pass --force to regenerate these synthetic files.")
    # Only these two known generated paths are ever written by this program.
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 2000
    time = np.arange(3 * sample_rate, dtype=np.float64) / sample_rate
    specs = [
        ("train_a", 4, 1.95, 2.70), ("train_b", 4, 2.10, None),
        ("train_c", 4, 1.80, 2.65), ("validation", 4, 2.00, None),
        ("test_a", 6, 2.20, None), ("test_b", 3, 1.70, 2.60),
        ("test_c", 5, 2.30, None),
    ]
    arrays = {}
    metadata = {}
    for run_number, (run_id, count, onset, recovery) in enumerate(specs):
        rng = np.random.default_rng(20261002 + run_number)
        state = (time >= onset).astype(np.int8)
        if recovery is not None:
            state[time >= recovery] = 0
        precursor = np.clip((time - (onset - .40)) / .40, 0, 1)
        precursor[time >= onset] = 0
        frequency = 35 + rng.uniform(-5, 5)
        columns = []
        for sensor in range(count):
            white = rng.normal(0, .025, len(time))
            colored = np.empty_like(white)
            colored[0] = white[0]
            for i in range(1, len(time)):
                colored[i] = .82 * colored[i - 1] + white[i]
            phase = rng.uniform(-.25, .25)
            gain = rng.uniform(.85, 1.15)
            normal = .06 * np.sin(2 * np.pi * (7 + .3 * sensor) * time + phase)
            synthetic_precursor = precursor * (
                .35 + .24 * np.sin(2 * np.pi * frequency * time + phase))
            unstart = state * (1.0 + .18 * np.sin(2 * np.pi * 18 * time + phase))
            signal = .5 + .02 * sensor + gain * (normal + synthetic_precursor + unstart) + colored
            columns.append(signal.astype(np.float32))
        arrays[f"{run_id}__time"] = time.copy()
        arrays[f"{run_id}__pressure"] = np.column_stack(columns)
        arrays[f"{run_id}__state"] = state
        sensor_ids = [f"S{i + 1:02d}" for i in range(count)]
        metadata[run_id] = {
            "sensor_ids": sensor_ids,
            "configurations": configurations(sensor_ids, 7300 + run_number),
            "source": "synthetic signal generator; not CFD",
            "pressure_unit": "arbitrary units",
            "onset_s": onset,
            "recovery_s": recovery,
        }
    config = {
        "dataset": "data/synthetic.npz",
        "data_kind": "synthetic",
        "description": "Synthetic smoke test only; not experimental measurements or CFD.",
        "seeds": [17, 29, 43],
        "settings": {"sample_rate": 2000, "window": 100, "stride": 20,
                     "horizon": .5, "confirmation": 3},
        "split": {"train": ["train_a", "train_b", "train_c"],
                  "validation": ["validation"],
                  "test": ["test_a", "test_b", "test_c"]},
        "runs": metadata,
        "training": {"epochs": 40, "patience": 8, "batch_size": 256,
                     "samples_per_class_per_run": 4000, "learning_rate": .001,
                     "weight_decay": .0001, "trees": 300, "min_samples_leaf": 5},
    }
    np.savez_compressed(dataset_path, **arrays)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"Generated {len(specs)} synthetic runs: {dataset_path}")
    print(f"Frozen split and sensor subsets: {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--force", action="store_true",
                        help="replace only config.json and data/synthetic.npz")
    args = parser.parse_args()
    generate(args.output_dir, force=args.force)
