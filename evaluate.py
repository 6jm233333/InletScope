import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from data import load_config, load_run
from models import build_model
from train import predict_all, save_json, sha256


def average_sensors(matrix, available, chosen):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[1] != len(available) or len(set(available)) != len(available):
        raise ValueError('Invalid probability matrix or sensor IDs')
    if not chosen:
        raise ValueError('No available observations: empty sensor set')
    if len(set(chosen)) != len(chosen) or not set(chosen) <= set(available):
        raise ValueError('Repeated or unknown selected sensor')
    columns = [available.index(s) for s in chosen]
    selected = matrix[:, columns]
    if not np.isfinite(selected).all() or np.any((selected < 0) | (selected > 1)):
        raise ValueError('Selected sensors contain missing or invalid probabilities; omit unavailable sensors')
    return selected.mean(axis=1)


def confirmed(probability, threshold, k=3):
    p = np.asarray(probability)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be a finite vector in [0,1]')
    if not isinstance(k, (int, np.integer)) or k < 1 or not 0 <= threshold <= 1:
        raise ValueError('Invalid confirmation rule')
    active = np.zeros(len(p), dtype=bool)
    count = 0
    for i, value in enumerate(p):
        count = count + 1 if value >= threshold else 0
        active[i] = count >= k
    return active


def episodes(mask):
    return int(np.count_nonzero(np.diff(np.r_[False, mask].astype(np.int8)) == 1))


def window_scores(y, p, eligible, threshold):
    y, p = y[eligible], p[eligible]
    if not len(y):
        return {'eligible_windows': 0, 'positive_windows': 0,
                'precision': None, 'recall': None, 'f1': None, 'ap': None, 'auroc': None}
    binary = p >= threshold
    return dict(eligible_windows=int(len(y)), positive_windows=int(y.sum()),
        precision=float(precision_score(y, binary, zero_division=0)),
        recall=float(recall_score(y, binary, zero_division=0)),
        f1=float(f1_score(y, binary, zero_division=0)),
        ap=float(average_precision_score(y, p)) if y.sum() else None,
        auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None)


def event_scores(run, p, threshold, k, task, horizon=0.5):
    if task not in ('detection', 'warning'):
        raise ValueError('Unknown monitoring task')
    if len(p) != len(run.time):
        raise ValueError('Probabilities must align with every output time')
    active = confirmed(p, threshold, k)
    rising = np.diff(np.r_[False, active].astype(np.int8)) == 1
    raw_time, state, output_time = run.raw_time, run.raw_state.astype(bool), run.time
    state_at_output = state[np.searchsorted(raw_time, output_time, side='right') - 1]
    # Initial unstart is left-censored: it has no observed new onset.
    starts = np.flatnonzero((~state[:-1]) & state[1:]) + 1
    normal_output = ~state_at_output
    normal_raw = ~state
    events = []
    for start in starts:
        onset = float(raw_time[start])
        after = np.flatnonzero(~state[start:])
        recovery = float(raw_time[start + after[0]]) if len(after) else None
        in_event = (output_time >= onset) & ((output_time < recovery) if recovery is not None else True)
        if task == 'detection':
            valid = rising & in_event
        elif task == 'warning':
            valid = rising & (output_time >= onset - horizon) & (output_time < onset)
            normal_output &= ~((output_time >= onset - horizon) & (output_time < onset))
            normal_raw &= ~((raw_time >= onset - horizon) & (raw_time < onset))
        else:
            raise ValueError(task)
        ids = np.flatnonzero(valid)
        first = float(output_time[ids[0]]) if len(ids) else None
        previous = np.flatnonzero(output_time < onset)
        event = dict(onset_s=onset, recovery_s=recovery, new_event_hit=bool(len(ids)),
            declaration_s=first,
            already_active_at_onset=bool(active[previous[-1]]) if len(previous) else False,
            event_overlap_hit=bool(np.any(active & in_event)) if task == 'detection' else None)
        event['delay_ms' if task == 'detection' else 'lead_ms'] = (
            float(1000 * (first - onset if task == 'detection' else onset - first)) if first is not None else None)
        events.append(event)
    if task == 'warning':
        normal_output &= output_time + horizon <= raw_time[-1] + 1e-9
        normal_raw &= raw_time + horizon <= raw_time[-1]
    durations = np.diff(raw_time, append=raw_time[-1] + np.median(np.diff(raw_time)))
    exposure = float(durations[normal_raw].sum())
    false_count = episodes(active & normal_output)
    return dict(events=events, event_count=len(events),
        event_hits=sum(e['new_event_hit'] for e in events),
        false_episodes=false_count,
        false_episodes_per_normal_min=false_count / (exposure / 60) if exposure > 0 else None,
        normal_exposure_s=exposure, alarm_fraction=float(active.mean()),
        initially_unstarted=bool(state[0]))


def evaluate(args):
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    dataset = config_path.parent / cfg['dataset']
    folder = Path(args.fit)
    fit = json.loads((folder / 'fit.json').read_text(encoding='utf-8'))
    if sha256(config_path) != fit['config_sha256'] or sha256(dataset) != fit['dataset_sha256']:
        raise ValueError('Config or dataset differs from the fitted version')
    torch.set_num_threads(4)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    name, task = fit['model'], fit['task']
    if not args.rescore:
        weights = folder / ('model.joblib' if name == 'rf' else 'model.pt')
        if sha256(weights) != fit['checkpoint_sha256']:
            raise ValueError('Checkpoint differs from the fitted version')
        if name == 'rf':
            # Load only checkpoints produced locally by train.py, never untrusted pickle files.
            model = joblib.load(weights)
            device = 'cpu'
        else:
            model = build_model(name).to(device)
            model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    rows = []
    for run_id in cfg['split']['test']:
        spec = cfg['runs'][run_id]
        run = load_run(dataset, run_id, task, cfg['settings'], spec['sensor_ids'])
        saved = folder / f'{run_id}_predictions.npz'
        if args.rescore:
            with np.load(saved, allow_pickle=False) as z:
                for key, expected in [('time', run.time), ('y', run.y), ('eligible', run.eligible),
                                      ('sensor_ids', np.array(run.sensor_ids))]:
                    if not np.array_equal(z[key], expected):
                        raise ValueError(f'Saved prediction metadata changed: {run_id}/{key}')
                matrix = z['probability']
                if matrix.shape != (len(run.time), len(run.sensor_ids)):
                    raise ValueError('Saved prediction shape mismatch')
        else:
            matrix = predict_all(model, name, run, **fit['normalization'], device=device)
            np.savez_compressed(saved, probability=matrix, time=run.time, y=run.y,
                                eligible=run.eligible, sensor_ids=np.array(run.sensor_ids))
        for configuration, sensors in spec['configurations'].items():
            p = average_sensors(matrix, run.sensor_ids, sensors)
            scores = dict(run=run_id, configuration=configuration, sensors=sensors,
                window=window_scores(run.y, p, run.eligible, fit['threshold']),
                alarm={str(k): event_scores(run, p, fit['threshold'], k, task, cfg['settings']['horizon'])
                       for k in sorted({1, cfg['settings']['confirmation']})})
            rows.append(scores)
            if configuration == 'all':
                alarm = scores['alarm'][str(cfg['settings']['confirmation'])]
                print(f'{run_id}: F1={scores["window"]["f1"]}; event hits={alarm["event_hits"]}; '
                      f'false episodes={alarm["false_episodes"]}')
    result = dict(data_kind=cfg['data_kind'], quick=fit['quick'], task=task, model=name,
                  seed=fit['seed'], threshold=fit['threshold'], evaluations=rows)
    save_json(folder / 'metrics.json', result)
    print(f'Saved {len(rows)} sensor-configuration evaluations to {folder / "metrics.json"}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config.json')
    p.add_argument('--fit', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    p.add_argument('--rescore', action='store_true', help='Recompute metrics from saved predictions without model inference')
    evaluate(p.parse_args())
