# InletScope

Minimal code for pressure-based detection and warning.

## Run

Python 3.11; CPU supported.

```sh
python -m pip install -r requirements.txt
python -B train.py --task detection --quick --device cpu
python -B evaluate.py --fit outputs/detection_inletscope_17 --device cpu
python -B train.py --task warning --quick --device cpu
python -B evaluate.py --fit outputs/warning_inletscope_17 --device cpu
python -B -m unittest -v test_core
```

`--quick` runs a small software check. Omit it for the full training settings.
Other models: `--model tcn|rf|single_scale|uniform_fusion`.
Results go to `outputs/`; use `--output` for a new run.

## Data

`data/synthetic.npz` contains seven synthetic pressure-signal runs.
These are test signals, not experimental measurements or CFD.
Real experimental data are not included.

Each run contains `<run>__time` (N), `<run>__pressure` (N,S), and
`<run>__state` (N; 0 normal, 1 unstarted).
Splits, sensor sets and settings are in `config.json`.
Regenerate the example with `python -B make_demo.py --force`.

## Files

- `models.py`: models and statistical features.
- `data.py`: causal windows and labels.
- `train.py`, `evaluate.py`: training and alarm evaluation.
- `make_demo.py`, `test_core.py`: data generation and tests.

MIT license. Do not upload generated `outputs/`.
