"""Exact metric equations and masked objectives; no threshold-based accuracy claims."""
import math
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from psdformer.losses import multitask_loss
from psdformer.metrics import classification_metrics, forecasting_metrics
from psdformer.model import PSDformer, PSDformerConfig

torch.set_num_threads(1)


class MetricTests(unittest.TestCase):
    def test_forecast_equations_distinguish_rse_from_rmse(self):
        target = np.array([1., 2., 3., 4.]).reshape(2, 2, 1)
        prediction = np.array([2., 4., 3., 7.]).reshape(2, 2, 1)
        result = forecasting_metrics(target, prediction)
        self.assertEqual(result['n_valid'], 4)
        self.assertAlmostEqual(result['mae'], 1.5)
        self.assertAlmostEqual(result['mse'], 3.5)
        self.assertAlmostEqual(result['rmse'], math.sqrt(3.5))
        self.assertAlmostEqual(result['rse'], math.sqrt(14 / (4 - 2)))
        self.assertAlmostEqual(result['r2'], 1 - 14 / 5)
        self.assertNotEqual(result['rse'], result['rmse'])

    def test_forecast_mask_counts_only_observed_targets(self):
        target = np.array([1., 2., 3., 4., np.nan, 90.])
        prediction = np.array([2., 4., 3., 7., np.nan, np.inf])
        mask = np.array([1, 1, 1, 1, 1, 0], dtype=bool)
        actual = forecasting_metrics(target, prediction, mask)
        expected = forecasting_metrics(target[:4], prediction[:4])
        self.assertEqual(actual, expected)

    def test_undefined_metrics_are_explicit(self):
        for size in (0, 1, 2):
            with self.subTest(size=size):
                result = forecasting_metrics(np.ones(size), np.ones(size))
                self.assertTrue(math.isnan(result['rse']))
                self.assertTrue(math.isnan(result['r2']))
        result = forecasting_metrics(np.full(4, np.nan), np.full(4, np.nan))
        self.assertEqual(result['n_valid'], 0)
        self.assertTrue(all(math.isnan(result[key]) for key in ('mae', 'mse', 'rmse', 'rse', 'r2')))

    def test_nonfinite_observed_prediction_and_bad_mask_are_rejected(self):
        with self.assertRaises(ValueError):
            forecasting_metrics(np.ones(3), np.array([1., np.nan, 1.]))
        with self.assertRaises(ValueError):
            forecasting_metrics(np.ones(3), np.ones(3), np.array([1, 2, 0]))
        with self.assertRaises(ValueError):
            forecasting_metrics(np.ones((3, 1)), np.ones(3))

    def test_classification_confusion_and_macro_metrics(self):
        labels = np.array([0, 1, 1, 2, -1])
        probabilities = np.array([[.8, .1, .1], [.1, .8, .1], [.7, .2, .1],
                                  [.1, .8, .1], [np.nan, np.nan, np.nan]])
        result = classification_metrics(labels, probabilities)
        self.assertEqual(result['n_labelled'], 4)
        self.assertEqual(result['confusion_matrix'], [[1, 0, 0], [1, 1, 0], [0, 1, 0]])
        self.assertEqual(result['support'], [1, 2, 1])
        self.assertAlmostEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['macro_precision'], 1 / 3)
        self.assertAlmostEqual(result['macro_recall'], .5)
        self.assertAlmostEqual(result['macro_f1'], (2 / 3 + .5 + 0) / 3)
        self.assertAlmostEqual(result['nll'], -np.log([.8, .8, .2, .1]).mean())

    def test_classification_invalid_labels_and_probabilities_are_rejected(self):
        for labels, probabilities in (
            ([.5], [[.5, .5]]), ([2], [[.5, .5]]), ([-2], [[.5, .5]]),
            ([0], [[np.nan, .5]]), ([0], [[.2, .2]]),
        ):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                classification_metrics(np.array(labels), np.array(probabilities))


class LossTests(unittest.TestCase):
    def test_joint_loss_matches_observed_mse_and_labelled_ce(self):
        forecast = torch.tensor([[[1.], [9.]], [[3.], [4.]]], requires_grad=True)
        logits = torch.tensor([[2., 0.], [0., 2.]], requires_grad=True)
        target = torch.tensor([[[2.], [float('nan')]], [[1.], [float('inf')]]])
        labels = torch.tensor([1, -1])
        result = multitask_loss({'forecast': forecast, 'logits': logits},
                                {'target': target, 'y': labels}, 'joint', 2., 3.)
        expected_ce = F.cross_entropy(logits[:1], labels[:1])
        self.assertEqual(result['n_forecast'], 2)
        self.assertEqual(result['n_classification'], 1)
        self.assertAlmostEqual(float(result['forecast'].detach()), 2.5)
        torch.testing.assert_close(result['total'], 2 * torch.tensor(2.5) + 3 * expected_ce)
        result['total'].backward()
        self.assertTrue(torch.isfinite(forecast.grad).all())
        self.assertEqual(float(forecast.grad[0, 1]), 0)
        self.assertEqual(float(forecast.grad[1, 1]), 0)
        torch.testing.assert_close(logits.grad[1], torch.zeros(2), rtol=0, atol=0)

    def test_forecast_only_requires_no_logits_or_class_labels(self):
        forecast = torch.ones(2, 3, 1, requires_grad=True)
        result = multitask_loss({'forecast': forecast}, {'target': torch.zeros_like(forecast)}, 'forecast')
        self.assertEqual(result['n_classification'], 0)
        result['total'].backward()
        self.assertTrue(torch.isfinite(forecast.grad).all())
        self.assertGreater(float(forecast.grad.abs().sum()), 0)

    def test_classification_only_does_not_inspect_bad_unused_targets(self):
        model = PSDformer(PSDformerConfig(input_dim=2, input_len=7, pred_len=1,
                           d_model=16, n_heads=4, dgc_growth_rate=4, dropout=0))
        output = model(torch.randn(2, 7, 2))
        result = multitask_loss(output, {'target': torch.full((2, 1, 1), float('nan')),
                                        'y': torch.tensor([0, 1])}, 'classification')
        self.assertEqual(result['n_forecast'], 0)
        self.assertTrue(torch.isfinite(result['total']))
        result['total'].backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(float(model.classifier.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.embedding.projection.weight.grad.abs().sum()), 0)

    def test_completely_missing_supervision_returns_finite_graph_zero(self):
        forecast = torch.full((2, 1, 1), float('nan'), requires_grad=True)
        logits = torch.full((2, 3), float('inf'), requires_grad=True)
        result = multitask_loss({'forecast': forecast, 'logits': logits},
                                {'target': torch.full_like(forecast, float('nan')),
                                 'y': torch.full((2,), -1, dtype=torch.long)}, 'joint')
        self.assertEqual(result['n_forecast'], 0)
        self.assertEqual(result['n_classification'], 0)
        self.assertEqual(float(result['total'].detach()), 0)
        result['total'].backward()
        self.assertTrue(torch.isfinite(forecast.grad).all())
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_zero_weight_disables_unused_task_and_invalid_weights_fail(self):
        forecast = torch.ones(2, 1, 1, requires_grad=True)
        result = multitask_loss({'forecast': forecast}, {'target': torch.zeros_like(forecast)},
                                'joint', 1., 0.)
        self.assertEqual(result['n_classification'], 0)
        for weights in ((0., 0.), (-1., 1.), (float('nan'), 1.), (1., float('inf'))):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                multitask_loss({'forecast': forecast}, {'target': torch.zeros_like(forecast)},
                               'joint', *weights)

    def test_epoch_loss_weights_by_observed_cells_not_batch_count(self):
        from psdformer.train import _epoch
        class IdentityForecast(torch.nn.Module):
            def forward(self, x, mask=None):
                return {'forecast': x}
        first = torch.tensor([[[1.]], [[2.]]])
        second = torch.tensor([[[3.]]])
        batches = [{'x': x, 'mask': torch.ones_like(x, dtype=torch.bool),
                    'target': torch.zeros_like(x)} for x in (first, second)]
        result = _epoch(IdentityForecast(), batches, torch.device('cpu'), 'forecast',
                        {'forecast_weight': 1., 'classification_weight': 0.})
        self.assertEqual(result['n_forecast'], 3)
        self.assertAlmostEqual(result['forecast'], 14 / 3)
        self.assertAlmostEqual(result['total'], 14 / 3)
        self.assertEqual(result['optimizer_steps'], 0)

    def test_nonfinite_observed_predictions_are_not_silently_dropped(self):
        forecast = torch.tensor([[[float('nan')]]], requires_grad=True)
        with self.assertRaises(ValueError):
            multitask_loss({'forecast': forecast}, {'target': torch.ones(1, 1, 1)}, 'forecast')


if __name__ == '__main__':
    unittest.main()
