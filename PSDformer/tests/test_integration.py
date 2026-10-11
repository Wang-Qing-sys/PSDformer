"""Executable CPU train/evaluate/predict checks, using only labelled synthetic fixtures."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from psdformer.data import load_npz, partition_indices
from psdformer.evaluate import load_checkpoint
from psdformer.metrics import forecasting_metrics
from psdformer.model import PSDformer, PSDformerConfig
from psdformer.preprocessing import StandardNormalizer

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def cli(*arguments, cwd=None):
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(ROOT) + os.pathsep + environment.get('PYTHONPATH', '')
    environment['OMP_NUM_THREADS'] = '1'
    result = subprocess.run([sys.executable, '-m', 'psdformer', *map(str, arguments)],
                            cwd=cwd or ROOT, env=environment, text=True, capture_output=True,
                            timeout=180)
    if result.returncode:
        raise AssertionError(f'CLI failed: {arguments}\n{result.stdout}\n{result.stderr}')
    return result


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.output = cls.directory / 'joint_demo'
        cls.config = json.loads((ROOT / 'configs' / 'smoke.json').read_text())
        cls.config['model'].update(input_len=13, pred_len=5, output_dim=2, encoder_levels=3)
        cls.config['training'].update(cpu_threads=1, batch_size=3, seed=37, epochs=2)
        cls.config['demo'] = {'subjects': 3, 'windows_per_subject': 3}
        cls.config_path = cls.directory / 'config.json'
        cls.config_path.write_text(json.dumps(cls.config))
        cls.demo_result = cli('demo', '--output', cls.output, '--config', cls.config_path,
                              '--device', 'cpu', cwd=cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_help_exposes_every_command(self):
        for command in ((), ('demo',), ('train',), ('evaluate',), ('predict',),
                        ('make-demo-data',), ('prepare-csv',)):
            with self.subTest(command=command):
                self.assertIn('usage:', cli(*command, '--help').stdout)

    def test_cli_demo_optimizes_and_selects_validation_checkpoint(self):
        for name in ('best.pt', 'last.pt', 'history.json', 'metrics.json', 'synthetic.npz',
                     'predictions.csv', 'split.json', 'config.json', 'demo_summary.json'):
            self.assertTrue((self.output / name).is_file(), name)
        history = json.loads((self.output / 'history.json').read_text())
        self.assertEqual(len(history), 2)
        for row in history:
            self.assertGreater(row['train']['optimizer_steps'], 0)
            self.assertGreater(row['train']['max_unclipped_gradient_norm'], 0)
            self.assertEqual(row['validation']['optimizer_steps'], 0)
            self.assertTrue(all(math.isfinite(row[partition]['total']) for partition in ('train', 'validation')))
        checkpoint = torch.load(self.output / 'best.pt', weights_only=True)
        torch.manual_seed(self.config['training']['seed'])
        initial = PSDformer(PSDformerConfig(**self.config['model'])).state_dict()
        changes = [name for name, value in initial.items()
                   if value.is_floating_point() and not torch.equal(value, checkpoint['model_state'][name])]
        self.assertTrue(changes)
        self.assertTrue(any(name.startswith('decoder_layers') for name in changes))
        self.assertTrue(any(name.startswith('dgc') for name in changes))
        self.assertTrue(any(name.startswith('classifier') for name in changes))
        self.assertTrue(checkpoint['optimizer_state']['state'])
        self.assertEqual(checkpoint['selection_metric'], 'val_total_loss')
        self.assertEqual(checkpoint['best_validation_value'], min(row['selection_value'] for row in history))
        self.assertTrue(checkpoint['synthetic_demo'])

    def test_checkpoint_contains_train_only_input_and_target_statistics(self):
        checkpoint = torch.load(self.output / 'best.pt', weights_only=True)
        data = load_npz(self.output / 'synthetic.npz')
        train = partition_indices(data['subject_ids'], checkpoint['split'], 'train')
        expected_x = StandardNormalizer.fit(data['x'][train], data['mask'][train])
        expected_target = StandardNormalizer.fit(data['target'][train])
        self.assertEqual(checkpoint['normalizer_fit_partition'], 'train')
        self.assertEqual(checkpoint['normalizers']['x'], expected_x.to_dict())
        self.assertEqual(checkpoint['normalizers']['target'], expected_target.to_dict())
        groups = [set(checkpoint['split'][part]) for part in ('train', 'val', 'test')]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_checkpoint_reload_predictions_are_identical_and_metrics_use_original_units(self):
        predictions_path = self.directory / 'reload.csv'
        cli('predict', '--data', self.output / 'synthetic.npz', '--checkpoint', self.output / 'best.pt',
            '--output', predictions_path, '--partition', 'test', '--batch-size', 3, '--threads', 1)
        self.assertEqual(predictions_path.read_bytes(), (self.output / 'predictions.csv').read_bytes())
        data = load_npz(self.output / 'synthetic.npz')
        model, checkpoint, normalizers = load_checkpoint(self.output / 'best.pt')
        test = partition_indices(data['subject_ids'], checkpoint['split'], 'test')
        x = normalizers['x'].transform(data['x'][test], data['mask'][test])
        with torch.no_grad():
            outputs = model(torch.from_numpy(x), torch.from_numpy(data['mask'][test]))
        raw_forecast = normalizers['target'].inverse_transform(outputs['forecast'].numpy())
        with predictions_path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(test))
        csv_forecast = np.array([[float(row[f'forecast_t{k + 1}_o{o}'])
                                  for k in range(5) for o in range(2)] for row in rows]).reshape(-1, 5, 2)
        np.testing.assert_array_equal(csv_forecast, raw_forecast)
        expected = forecasting_metrics(data['target'][test], raw_forecast)
        report = json.loads((self.output / 'metrics.json').read_text())
        self.assertEqual(set(report['subject_ids']), set(checkpoint['split']['test']))
        self.assertEqual(report['partition'], 'test')
        self.assertTrue(report['synthetic_demo'])
        for key, value in expected.items():
            self.assertAlmostEqual(report['metrics']['forecasting'][key], value, delta=1e-5)
        for row in rows:
            self.assertAlmostEqual(sum(float(row[f'prob_class_{c}']) for c in range(3)), 1., places=6)

    def test_new_unlabelled_data_predicts_but_cannot_impersonate_saved_test_partition(self):
        data = load_npz(self.output / 'synthetic.npz')
        incoming = self.directory / 'new_unlabelled.npz'
        np.savez(incoming, x=data['x'][:2], mask=data['mask'][:2], subject_ids=np.array(['new_a', 'new_b']))
        output = self.directory / 'new_predictions.npz'
        cli('predict', '--data', incoming, '--checkpoint', self.output / 'best.pt',
            '--output', output, '--partition', 'all', '--threads', 1)
        with np.load(output, allow_pickle=False) as predictions:
            self.assertEqual(predictions['forecast'].shape, (2, 5, 2))
            self.assertEqual(predictions['probs'].shape, (2, 3))
            self.assertTrue(np.isfinite(predictions['forecast']).all())
        with self.assertRaisesRegex(AssertionError, 'Data differs from the training NPZ'):
            cli('evaluate', '--data', incoming, '--checkpoint', self.output / 'best.pt',
                '--output', self.directory / 'invalid_metrics.json', '--partition', 'test', '--threads', 1)

    def test_forecast_and_classification_only_cli_training_with_missing_other_task(self):
        for task in ('forecast', 'classification'):
            with self.subTest(task=task):
                config = json.loads(json.dumps(self.config))
                config['model'].update(input_len=7, input_dim=2, pred_len=1, output_dim=2,
                                       num_classes=None if task == 'forecast' else 3)
                config['training'].update(task=task, epochs=1, batch_size=2,
                    selection_metric='val_forecast_loss' if task == 'forecast' else 'val_macro_f1')
                config_path = self.directory / f'{task}.json'
                config_path.write_text(json.dumps(config))
                rng = np.random.default_rng(22)
                arrays = {'x': rng.normal(size=(6, 7, 2)).astype(np.float32),
                          'subject_ids': np.repeat(['s0', 's1', 's2'], 2)}
                if task == 'forecast':
                    arrays['target'] = (10 + 3 * rng.normal(size=(6, 1, 2))).astype(np.float32)
                else:
                    arrays['y'] = np.array([0, 1, 0, 2, 1, 2], dtype=np.int64)
                data_path, output = self.directory / f'{task}.npz', self.directory / task
                np.savez(data_path, **arrays)
                cli('train', '--data', data_path, '--config', config_path, '--output', output, '--device', 'cpu')
                checkpoint = torch.load(output / 'best.pt', weights_only=True)
                self.assertEqual(checkpoint['task'], task)
                self.assertGreater(checkpoint['global_step'], 0)
                history = json.loads((output / 'history.json').read_text())
                self.assertTrue(math.isfinite(history[0]['train']['total']))
                cli('evaluate', '--data', data_path, '--checkpoint', output / 'best.pt',
                    '--output', output / 'metrics.json', '--threads', 1)
                prediction_path = output / 'predictions.npz'
                cli('predict', '--data', data_path, '--checkpoint', output / 'best.pt',
                    '--output', prediction_path, '--threads', 1)
                with np.load(prediction_path, allow_pickle=False) as predictions:
                    self.assertEqual('forecast' in predictions, task == 'forecast')
                    self.assertEqual('probs' in predictions, task == 'classification')
                metrics = json.loads((output / 'metrics.json').read_text())['metrics']
                self.assertEqual('forecasting' in metrics, task == 'forecast')
                self.assertEqual('classification' in metrics, task == 'classification')


if __name__ == '__main__':
    unittest.main()
