"""Local, data-independent checks of the monitoring protocol."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data import WindowedRun, load_config, load_run
from evaluate import average_sensors, confirmed, event_scores
from make_demo import generate
from models import build_model, statistical_features
from train import choose_threshold, predict_all, sampled_epoch, source_stats

ROOT = Path(__file__).resolve().parent


def signal(task='warning', onset=1.0, recovery=1.6, length=4000):
    t = np.arange(length) / 2000
    state = (t >= onset) & (t < recovery)
    x = np.column_stack((np.sin(40 * t), np.cos(50 * t))).astype(np.float32)
    return WindowedRun('example', t, x, state, task)


class DataTests(unittest.TestCase):
    def test_causal_window_and_no_future_input(self):
        r = signal()
        np.testing.assert_array_equal(r.windows[0], r.values[:100].T)
        self.assertEqual(r.time[0], r.raw_time[99])
        x = r.values.copy(); x[100:] += 100
        changed = WindowedRun('changed', r.raw_time, x, r.raw_state, 'warning')
        np.testing.assert_array_equal(r.windows[0], changed.windows[0])

    def test_warning_exact_right_boundary_and_current_onset(self):
        # First window ends at .0495; its right horizon boundary is .5495.
        r = signal(onset=.5495)
        self.assertTrue(r.eligible[0]); self.assertEqual(r.y[0], 1)
        r = signal(onset=.0495)
        self.assertFalse(r.eligible[0]); self.assertEqual(r.y[0], 0)

    def test_incomplete_future_and_recovery(self):
        r = signal(recovery=1.3)
        self.assertTrue(np.any(r.eligible & (r.time > 1.35)))
        self.assertFalse(np.any(r.eligible[r.time > r.raw_time[-1] - .5 + 1e-9]))
        short_future = signal(length=200)
        self.assertFalse(np.any(short_future.eligible))
        with self.assertRaises(ValueError):
            signal(length=99)

    def test_initial_state_is_not_new_onset(self):
        r = signal(onset=0, recovery=5)
        self.assertEqual(len(r.onsets), 0)
        self.assertFalse(r.y.any())
        self.assertEqual(event_scores(r, np.ones(len(r.time)), .5, 3, 'warning')['event_count'], 0)

    def test_time_column_nonfinite_and_duplicate_ids(self):
        r = signal()
        for pressure in (np.column_stack((r.raw_time, r.values[:, 0])), r.values * np.nan):
            with self.assertRaises(ValueError):
                WindowedRun('invalid', r.raw_time, pressure, r.raw_state, 'detection')
        with self.assertRaises(ValueError):
            WindowedRun('invalid', r.raw_time, r.values, r.raw_state, 'detection', sensor_ids=['A', 'A'])

    def test_run_isolation_and_frozen_subsets(self):
        cfg = load_config(ROOT / 'config.json')
        for r in cfg['split']['test']:
            subsets = cfg['runs'][r]['configurations']
            for count in (1, 2, 4, 8):
                selected = [tuple(sorted(v)) for k, v in subsets.items() if k.startswith(f'count_{count}_')]
                self.assertEqual(len(set(selected)), len(selected))
        bad = copy.deepcopy(cfg); bad['split']['test'][0] = bad['split']['train'][0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.json'; path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_bundled_data_generator_matches_arrays(self):
        cfg = load_config(ROOT / 'config.json')
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp)
            with np.load(ROOT / cfg['dataset']) as a, np.load(Path(tmp) / cfg['dataset']) as b:
                self.assertEqual(a.files, b.files)
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(cfg, load_config(Path(tmp) / 'config.json'))
            with self.assertRaises(FileExistsError):
                generate(tmp)


class ModelTests(unittest.TestCase):
    def test_model_counts_and_forward(self):
        torch.set_num_threads(2)
        for name, count in [('inletscope', 222085), ('single_scale', 86467),
                            ('uniform_fusion', 219906), ('tcn', 66626)]:
            model = build_model(name).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            with torch.no_grad():
                logits = model(torch.zeros(2, 100, 1))
            self.assertEqual(tuple(logits.shape), (2, 2))
            self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(statistical_features(np.zeros((3, 100))).shape, (3, 8))

    def test_source_moments_and_balanced_draws(self):
        cfg = load_config(ROOT / 'config.json')
        runs = [load_run(ROOT / cfg['dataset'], r, 'warning', cfg['settings']) for r in cfg['split']['train']]
        mean, std = source_stats(runs)
        expected = np.mean([r.values.astype(float).mean(axis=0).mean() for r in runs])
        self.assertAlmostEqual(mean, expected)
        x, y, counts = sampled_epoch(runs, np.random.default_rng(17), mean, std, 16)
        self.assertEqual(x.shape, (96, 100)); self.assertEqual(int(y.sum()), 48)
        for run_counts in counts.values():
            self.assertEqual([sum(x.values()) for x in run_counts.values()], [16, 16])

    def test_inference_does_not_read_labels_or_eligibility(self):
        torch.set_num_threads(2)
        r = signal()
        model = build_model('tcn').eval()
        original = predict_all(model, 'tcn', r, 0, 1)
        r.y[:] = 1 - r.y; r.eligible[:] = False; r.raw_state[:] = 1
        np.testing.assert_array_equal(original, predict_all(model, 'tcn', r, 0, 1))

    def test_threshold_tie_uses_highest(self):
        self.assertEqual(choose_threshold(np.array([0, 1]), np.array([0., 1.])), .99)


class AlarmTests(unittest.TestCase):
    def test_sensor_order_singleton_and_missing(self):
        p = np.array([[.1, .5, .9], [.4, .3, .2]])
        names = ['A', 'B', 'C']
        np.testing.assert_allclose(average_sensors(p, names, names), average_sensors(p[:, ::-1], names[::-1], names))
        np.testing.assert_array_equal(average_sensors(p, names, ['B']), p[:, 1])
        p[:, 2] = np.nan
        np.testing.assert_allclose(average_sensors(p, names, ['A', 'B']), p[:, :2].mean(1))
        for chosen in ([], ['A', 'A'], ['missing'], ['C']):
            with self.assertRaises(ValueError):
                average_sensors(p, names, chosen)

    def test_confirmation_and_reset(self):
        np.testing.assert_array_equal(confirmed([1, 1, 1, 0, 1, 1, 1], .5), [0, 0, 1, 0, 0, 0, 1])
        r = signal(onset=10)
        with self.assertRaises(ValueError):
            event_scores(r, np.zeros(len(r.time)), .5, 3, 'invalid')
        with self.assertRaises(ValueError):
            event_scores(r, np.zeros(len(r.time) - 1), .5, 3, 'warning')

    def test_always_normal_and_always_alarm(self):
        r = signal()
        off = event_scores(r, np.zeros(len(r.time)), .5, 3, 'warning')
        on = event_scores(r, np.ones(len(r.time)), .5, 3, 'warning')
        self.assertFalse(off['events'][0]['new_event_hit'])
        self.assertIsNone(off['events'][0]['lead_ms'])
        self.assertFalse(on['events'][0]['new_event_hit'])
        self.assertGreater(on['false_episodes'], 0)
        detection = event_scores(r, np.ones(len(r.time)), .5, 3, 'detection')
        self.assertTrue(detection['events'][0]['event_overlap_hit'])
        self.assertFalse(detection['events'][0]['new_event_hit'])
        self.assertEqual(detection['false_episodes'], 2)  # before onset and after recovery

    def test_valid_warning_and_confirmation_crossing_onset(self):
        r = signal()
        near = (r.time >= .8).astype(float)
        self.assertTrue(event_scores(r, near, .5, 3, 'warning')['events'][0]['new_event_hit'])
        crossing = (r.time >= .989).astype(float)
        self.assertFalse(event_scores(r, crossing, .5, 3, 'warning')['events'][0]['new_event_hit'])

    def test_multiple_events(self):
        r = signal()
        state = ((r.raw_time >= .6) & (r.raw_time < .8)) | ((r.raw_time >= 1.3) & (r.raw_time < 1.6))
        r = WindowedRun('two_events', r.raw_time, r.values, state, 'warning')
        p = (((r.time >= .4) & (r.time < .7)) | ((r.time >= 1.1) & (r.time < 1.4))).astype(float)
        scores = event_scores(r, p, .5, 3, 'warning')
        self.assertEqual(scores['event_count'], 2); self.assertEqual(scores['event_hits'], 2)


if __name__ == '__main__':
    unittest.main()
