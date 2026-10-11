"""Independent checks of the manuscript's two axes, routers, hierarchy and dense reuse."""
import math
import unittest

import torch

from psdformer.blocks import (
    MultiDimensionalSegmentEmbedding, SegmentMerging,
    TwoStageDependencyExtractor, DenseGraphConvolution, fixed_2d_positions,
)
from psdformer.model import PSDformer, PSDformerConfig

torch.set_num_threads(1)


class BlockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_embedding_segments_each_variable_and_zero_pads_last_segment(self):
        layer = MultiDimensionalSegmentEmbedding(input_dim=2, input_len=5, segment_len=3, d_model=3)
        with torch.no_grad():
            layer.projection.weight.copy_(torch.eye(3))
            layer.projection.bias.zero_()
        x = torch.tensor([[[1., 10.], [2., 20.], [3., 30.], [4., 40.], [5., 50.]]])
        tokens = layer(x) - layer.position_encoding
        expected = torch.tensor([[[[1., 2., 3.], [4., 5., 0.]],
                                  [[10., 20., 30.], [40., 50., 0.]]]])
        torch.testing.assert_close(tokens, expected)
        self.assertFalse(layer.position_encoding.requires_grad)
        self.assertIn('position_encoding', dict(layer.named_buffers()))

    def test_fixed_positions_distinguish_both_axes_and_odd_widths(self):
        for width in (1, 3, 16):
            with self.subTest(width=width):
                positions = fixed_2d_positions(3, 5, width)
                self.assertEqual(positions.shape, (1, 3, 5, width))
                self.assertFalse(torch.equal(positions[:, 0, 0], positions[:, 1, 0]))
                self.assertFalse(torch.equal(positions[:, 0, 0], positions[:, 0, 1]))
                torch.testing.assert_close(positions, fixed_2d_positions(3, 5, width), rtol=0, atol=0)

    def test_temporal_and_router_attention_axes_and_gradients(self):
        layer = TwoStageDependencyExtractor(d_model=16, n_heads=4, segments=5,
                                             router_count=2, ff_multiplier=2, dropout=0)
        observed = {}
        handles = []
        for name in ('temporal_attention', 'router_collect', 'router_distribute'):
            def record(module, inputs, name=name):
                observed[name] = [tuple(value.shape) for value in inputs[:3]]
            handles.append(getattr(layer, name).register_forward_pre_hook(record))
        x = torch.randn(2, 3, 5, 16, requires_grad=True)
        output = layer(x)
        for handle in handles:
            handle.remove()
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(observed['temporal_attention'], [(6, 5, 16)] * 3)
        self.assertEqual(observed['router_collect'], [(10, 2, 16), (10, 3, 16), (10, 3, 16)])
        self.assertEqual(observed['router_distribute'], [(10, 3, 16), (10, 2, 16), (10, 2, 16)])
        # Nonuniform coefficients avoid the constant squared norm induced by LayerNorm.
        (output * torch.randn_like(output)).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertIsNotNone(layer.routers.grad)
        self.assertTrue(torch.isfinite(layer.routers.grad).all())
        self.assertGreater(float(layer.routers.grad.abs().sum()), 0)

    def test_adjacent_segment_merging_and_odd_final_replication(self):
        layer = SegmentMerging(2)
        with torch.no_grad():
            layer.projection.weight.copy_(torch.cat((torch.eye(2), 10 * torch.eye(2)), dim=1))
            layer.projection.bias.zero_()
        x = torch.tensor([[[[1., 2.], [3., 4.], [5., 6.]]]])
        expected = torch.tensor([[[[31., 42.], [55., 66.]]]])
        torch.testing.assert_close(layer(x), expected)

    def test_dense_stage_uses_two_blocks_and_all_preceding_features(self):
        stage = DenseGraphConvolution(growth_rate=4, layers_per_block=2, pool_size=2)
        self.assertEqual(len(stage.blocks), 2)
        convolution_inputs = []
        handles = []
        for block in stage.blocks:
            for layer in block.layers:
                handles.append(layer[0].register_forward_pre_hook(
                    lambda module, args: convolution_inputs.append(tuple(args[0].shape))))
        output = stage(torch.randn(2, 3, 5, requires_grad=True))
        for handle in handles:
            handle.remove()
        self.assertEqual(convolution_inputs, [(2, 1, 3, 5), (2, 5, 3, 5),
                                              (2, 9, 3, 3), (2, 13, 3, 3)])
        self.assertEqual(output.shape, (2, 17, 3, 2))
        output.square().sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in stage.parameters()))

    def test_dense_pooling_preserves_nonempty_short_horizons(self):
        for horizon in (1, 2, 3, 5):
            for pool in (1, 2, 4):
                with self.subTest(horizon=horizon, pool=pool):
                    stage = DenseGraphConvolution(2, 1, pool)
                    output = stage(torch.randn(2, 3, horizon))
                    expected = math.ceil(math.ceil(horizon / pool) / pool)
                    self.assertEqual(output.shape[-1], expected)
                    self.assertTrue(torch.isfinite(output).all())

    def test_decoder_scale_predictions_are_summed_then_passed_into_dgc(self):
        config = PSDformerConfig(input_len=17, pred_len=7, input_dim=3,
                                d_model=16, n_heads=4, encoder_levels=4,
                                dgc_growth_rate=4, dropout=0)
        model = PSDformer(config).eval()
        self.assertEqual(model.encoder_segment_counts, [5, 3, 2, 1])
        captured = []
        handle = model.dgc.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        with torch.no_grad():
            output = model(torch.randn(2, 17, 3))
        handle.remove()
        self.assertEqual(output['scale_forecasts'].shape, (2, 4, 7, 3))
        torch.testing.assert_close(output['edm_forecast'], output['scale_forecasts'].sum(dim=1), rtol=0, atol=0)
        torch.testing.assert_close(captured[0], output['edm_forecast'].transpose(1, 2), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
