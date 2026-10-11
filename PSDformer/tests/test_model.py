"""Architecture tests against tensor contracts, without accuracy assumptions."""
from dataclasses import asdict
import unittest

import torch

from psdformer.model import PSDformer, PSDformerConfig

torch.set_num_threads(1)


def small_config(**changes):
    values = dict(input_dim=5, input_len=13, pred_len=5, output_dim=2,
                  num_classes=3, d_model=16, n_heads=4, segment_len=4,
                  router_count=2, tde_layers=1, encoder_levels=3,
                  dropout=0.0, ff_multiplier=2, dgc_growth_rate=4,
                  dgc_layers_per_block=2, dgc_pool_size=2)
    values.update(changes)
    return PSDformerConfig(**values)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def assert_valid_forward_backward(self, config, batch_size=2):
        model = PSDformer(config)
        x = torch.randn(batch_size, config.input_len, config.input_dim, requires_grad=True)
        output = model(x)
        self.assertEqual(output['forecast'].shape, (batch_size, config.pred_len, config.output_dim))
        self.assertEqual(output['edm_forecast'].shape, (batch_size, config.pred_len, config.input_dim))
        self.assertEqual(output['features'].shape, (batch_size, config.d_model))
        loss = output['forecast'].square().mean()
        if config.num_classes:
            self.assertEqual(output['logits'].shape, (batch_size, config.num_classes))
            loss = loss + output['logits'].square().mean()
        else:
            self.assertNotIn('logits', output)
        for key, value in output.items():
            if torch.is_tensor(value):
                self.assertTrue(torch.isfinite(value).all(), key)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()), 0)
        for group in ('embedding', 'pre_encoder', 'encoder_tdes', 'decoder_layers', 'dgc',
                      'forecast_head'):
            parameters = [parameter for name, parameter in model.named_parameters()
                          if name.startswith(group)]
            self.assertTrue(parameters, f'Missing expected module: {group}')
            active = [parameter.grad for parameter in parameters if parameter.grad is not None]
            self.assertTrue(active, f'No gradient reached {group}')
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in active), group)
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in active), 0, group)
        return model

    def test_default_model_forward_and_backward(self):
        self.assert_valid_forward_backward(PSDformerConfig())

    def test_odd_input_segments_prediction_horizon_and_output_channels(self):
        self.assert_valid_forward_backward(small_config())

    def test_single_sample_single_channel_single_step(self):
        self.assert_valid_forward_backward(small_config(input_dim=1, input_len=1,
            pred_len=1, output_dim=1, segment_len=4), batch_size=1)

    def test_classification_head_is_optional(self):
        for num_classes in (None, 0):
            with self.subTest(num_classes=num_classes):
                self.assert_valid_forward_backward(small_config(num_classes=num_classes))

    def test_masked_hidden_values_do_not_change_prediction(self):
        model = PSDformer(small_config()).eval()
        x = torch.randn(2, 13, 5)
        observed = torch.rand_like(x) > 0.3
        hidden = x.clone()
        hidden[~observed] = 1e9
        with torch.no_grad():
            expected = model(x, mask=observed)
            actual = model(hidden, mask=observed)
        for key in ('forecast', 'edm_forecast', 'logits', 'features'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    def test_nonfinite_values_are_sanitized_as_unobserved(self):
        model = PSDformer(small_config()).eval()
        x = torch.randn(2, 13, 5)
        x[0, 0, 0], x[0, 4, 2], x[1, 7, 3] = float('nan'), float('inf'), -float('inf')
        observed = torch.isfinite(x)
        clean = torch.where(observed, x, torch.zeros_like(x))
        with torch.no_grad():
            expected = model(clean, mask=observed)
            actual = model(x)
            all_missing = model(torch.full_like(x, float('nan')))
        for key in ('forecast', 'edm_forecast', 'logits', 'features'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
            self.assertTrue(torch.isfinite(all_missing[key]).all(), key)

    def test_masked_entries_receive_zero_input_gradient(self):
        model = PSDformer(small_config())
        x = torch.randn(2, 13, 5, requires_grad=True)
        observed = torch.rand_like(x) > 0.4
        model(x, mask=observed)['forecast'].square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.equal(x.grad[~observed], torch.zeros_like(x.grad[~observed])))
        self.assertGreater(float(x.grad[observed].abs().sum()), 0)

    def test_batches_do_not_exchange_information(self):
        model = PSDformer(small_config()).eval()
        x = torch.randn(3, 13, 5)
        with torch.no_grad():
            batch = model(x)
            separate = [model(x[i:i + 1]) for i in range(len(x))]
        for key in ('forecast', 'edm_forecast', 'logits', 'features'):
            joined = torch.cat([output[key] for output in separate], dim=0)
            torch.testing.assert_close(batch[key], joined, atol=2e-6, rtol=2e-5)

    def test_serialization_roundtrip_has_identical_predictions(self):
        config = small_config()
        original = PSDformer(config).eval()
        restored = PSDformer(PSDformerConfig(**asdict(config))).eval()
        restored.load_state_dict(original.state_dict())
        x = torch.randn(2, config.input_len, config.input_dim)
        with torch.no_grad():
            expected, actual = original(x), restored(x)
        for key in ('forecast', 'edm_forecast', 'logits', 'features'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    def test_adam_step_really_changes_parameters(self):
        model = PSDformer(small_config())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer.zero_grad()
        output = model(torch.randn(2, 13, 5))
        loss = output['forecast'].square().mean() + output['logits'].square().mean()
        loss.backward()
        optimizer.step()
        changed = [name for name, p in model.named_parameters()
                   if not torch.equal(before[name], p)]
        self.assertTrue(changed)
        self.assertTrue(any(name.startswith('dgc') for name in changed))
        self.assertTrue(any(name.startswith('decoder_layers') for name in changed))

    def test_invalid_shapes_fail_clearly(self):
        model = PSDformer(small_config())
        for x in (torch.randn(13, 5), torch.randn(2, 12, 5), torch.randn(2, 13, 4)):
            with self.subTest(shape=tuple(x.shape)), self.assertRaises((ValueError, RuntimeError)):
                model(x)
        with self.assertRaises((ValueError, RuntimeError)):
            model(torch.randn(2, 13, 5), mask=torch.ones(2, 13, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
