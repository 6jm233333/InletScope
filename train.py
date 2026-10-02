"""Train a shared single-sensor classifier; select settings on validation only."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score

from data import load_config, load_run
from models import build_model, statistical_features


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def source_stats(runs):
    """Equal weight per run, then per sensor; no target-run normalization."""
    means = [r.values.astype(np.float64).mean(axis=0).mean() for r in runs]
    squares = [(r.values.astype(np.float64) ** 2).mean(axis=0).mean() for r in runs]
    mean = float(np.mean(means))
    return mean, float(np.sqrt(max(np.mean(squares) - mean * mean, 1e-12)))


def sampled_epoch(runs, rng, mean, std, draws):
    features, labels, counts = [], [], {}
    for run in runs:
        counts[run.run] = {}
        for klass in (0, 1):
            choices = np.flatnonzero(run.eligible & (run.y == klass))
            if not len(choices):
                raise ValueError(f'No eligible class {klass} windows in training run {run.run}')
            idx = rng.choice(choices, draws, replace=True)
            sensors = rng.integers(len(run.sensor_ids), size=draws)
            features.append(run.windows[idx, sensors, :])
            labels.append(np.full(draws, klass, dtype=np.int64))
            counts[run.run][str(klass)] = dict(zip(
                run.sensor_ids, np.bincount(sensors, minlength=len(run.sensor_ids)).tolist()))
    x = np.asarray((np.concatenate(features) - mean) / std, dtype=np.float32)
    y = np.concatenate(labels)
    order = rng.permutation(len(y))
    return np.ascontiguousarray(x[order]), y[order], counts


def predict_all(model, name, run, mean, std, device='cpu', batch_size=1024):
    """Infer every window. Neither labels nor eligibility gate online outputs."""
    w, s, length = run.windows.shape
    flat = run.windows.reshape(w * s, length)
    outputs = []
    if name != 'rf':
        model.eval()
    for start in range(0, len(flat), batch_size):
        batch = np.asarray((flat[start:start + batch_size] - mean) / std, dtype=np.float32)
        if name == 'rf':
            outputs.append(model.predict_proba(statistical_features(batch))[:, 1])
        else:
            with torch.inference_mode():
                x = torch.from_numpy(batch[:, :, None]).to(device)
                outputs.append(torch.softmax(model(x), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(outputs).reshape(w, s).astype(np.float32)


def choose_threshold(y, probability):
    grid = np.arange(1, 100) / 100
    return float(max(grid, key=lambda t: (f1_score(y, probability >= t, zero_division=0), t)))


def train(args):
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    dataset = (config_path.parent / cfg['dataset']).resolve()
    folder = Path(args.output or f'outputs/{args.task}_{args.model}_{args.seed}')
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'Output is not empty: {folder}. Choose a new --output directory.')
    if args.quick and cfg.get('data_kind') != 'synthetic':
        raise ValueError('--quick is for synthetic software checks only')
    if args.seed not in cfg['seeds']:
        raise ValueError('Choose a seed listed in config.json')
    settings = dict(cfg['training'])
    if args.quick:
        settings.update(epochs=2, samples_per_class_per_run=64, batch_size=64, trees=30)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if args.model == 'rf':
        device = 'cpu'
    def read(run_id):
        return load_run(dataset, run_id, args.task, cfg['settings'], cfg['runs'][run_id]['sensor_ids'])
    train_runs = [read(r) for r in cfg['split']['train']]
    if len(cfg['split']['validation']) != 1:
        raise ValueError('This protocol uses exactly one validation run')
    val = read(cfg['split']['validation'][0])
    if len(np.unique(val.y[val.eligible])) != 2:
        raise ValueError('Validation requires both eligible classes')
    mean, std = source_stats(train_runs)
    rng = np.random.default_rng(args.seed)
    history, sampling = [], []
    started = time.perf_counter()
    folder.mkdir(parents=True, exist_ok=True)
    if args.model == 'rf':
        x, y, counts = sampled_epoch(train_runs, rng, mean, std, settings['samples_per_class_per_run'])
        sampling.append(counts)
        model = RandomForestClassifier(n_estimators=settings['trees'],
            min_samples_leaf=settings['min_samples_leaf'], class_weight='balanced',
            random_state=args.seed, n_jobs=4)
        model.fit(statistical_features(x), y)
        checkpoint = folder / 'model.joblib'
        joblib.dump(model, checkpoint, compress=3)
    else:
        model = build_model(args.model).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                      weight_decay=settings['weight_decay'])
        best, stale, best_state = -1.0, 0, None
        for epoch in range(settings['epochs']):
            x, y, counts = sampled_epoch(train_runs, rng, mean, std, settings['samples_per_class_per_run'])
            sampling.append(counts)
            model.train()
            total = 0.0
            # Equal-sized chunks match the original experiment's batch construction.
            batches = np.array_split(np.arange(len(y)), int(np.ceil(len(y) / settings['batch_size'])))
            for ids in batches:
                bx = torch.from_numpy(x[ids, :, None]).to(device)
                by = torch.from_numpy(y[ids]).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(bx), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(ids)
            probabilities = predict_all(model, args.model, val, mean, std, device)
            score = float(average_precision_score(val.y[val.eligible], probabilities.mean(axis=1)[val.eligible]))
            row = {'epoch': epoch + 1, 'loss': total / len(y), 'validation_ap': score}
            history.append(row)
            print(json.dumps(row), flush=True)
            if score > best + 1e-5:
                best, stale = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= settings['patience']:
                break
        if best_state is None:
            raise RuntimeError('No valid checkpoint was selected')
        model.load_state_dict(best_state)
        checkpoint = folder / 'model.pt'
        torch.save(best_state, checkpoint)
    p = predict_all(model, args.model, val, mean, std, device)
    pooled = p.mean(axis=1)[val.eligible]
    threshold = choose_threshold(val.y[val.eligible], pooled)
    np.savez_compressed(folder / 'validation_predictions.npz', probability=p, time=val.time,
                        y=val.y, eligible=val.eligible, sensor_ids=np.array(val.sensor_ids))
    result = dict(task=args.task, model=args.model, seed=args.seed,
        data_kind=cfg['data_kind'], quick=args.quick, settings=settings,
        config_sha256=sha256(config_path), dataset_sha256=sha256(dataset),
        checkpoint_sha256=sha256(checkpoint), normalization={'mean': mean, 'std': std},
        threshold=threshold, source_runs=cfg['split']['train'], validation_run=val.run,
        validation_ap=float(average_precision_score(val.y[val.eligible], pooled)),
        validation_f1=float(f1_score(val.y[val.eligible], pooled >= threshold, zero_division=0)),
        history=history, sampled_sensor_counts=sampling, training_seconds=time.perf_counter()-started)
    save_json(folder / 'fit.json', result)
    print(f'Saved {folder}; validation threshold={threshold:.2f}. No test run used in fitting.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config.json')
    p.add_argument('--task', choices=['detection', 'warning'], default='detection')
    p.add_argument('--model', choices=['inletscope', 'tcn', 'rf', 'single_scale', 'uniform_fusion'], default='inletscope')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    p.add_argument('--quick', action='store_true', help='Two-epoch synthetic smoke test, not a paper experiment')
    p.add_argument('--output', help='New directory for local generated outputs')
    train(p.parse_args())
