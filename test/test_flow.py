from __future__ import annotations

import unittest

import torch

from lpap.flow import DilatedConvFlow1d, flow_matching_loss, integrate_euler_midpoint_time
from lpap.image_to_energy_training import ImageToEnergyTimeConfig, sample_image_to_energy_time


class FlowTest(unittest.TestCase):
    def test_flow_model_shape_and_gradients(self) -> None:
        model = DilatedConvFlow1d(
            sequence_length=16,
            width=8,
            time_dim=8,
            dilation_cycles=1,
            dilations=(1, 2),
        )
        values = torch.randn(3, 1, 16)
        time = torch.rand(3)

        output = model(values, time)
        output.square().mean().backward()

        self.assertEqual(output.shape, values.shape)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_flow_matching_loss_is_scalar_and_finite(self) -> None:
        model = DilatedConvFlow1d(
            sequence_length=8,
            width=8,
            time_dim=8,
            dilation_cycles=1,
            dilations=(1,),
        )
        start = torch.randn(2, 1, 8)
        end = torch.randn(2, 1, 8)
        time = torch.tensor([0.25, 0.75])

        loss, metrics = flow_matching_loss(model, start, end, time)

        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics.loss, metrics.velocity_mse)

    def test_time_sampling_respects_eps(self) -> None:
        config = ImageToEnergyTimeConfig(distribution="uniform", eps=0.1)

        time = sample_image_to_energy_time(
            batch_size=1024,
            config=config,
            generator=torch.Generator().manual_seed(123),
        )

        self.assertGreaterEqual(float(time.min()), 0.1)
        self.assertLessEqual(float(time.max()), 0.9)

    def test_beta_time_sampling_uses_generator(self) -> None:
        config = ImageToEnergyTimeConfig(distribution="beta", eps=0.01)

        time_a = sample_image_to_energy_time(
            batch_size=16,
            config=config,
            generator=torch.Generator().manual_seed(123),
        )
        time_b = sample_image_to_energy_time(
            batch_size=16,
            config=config,
            generator=torch.Generator().manual_seed(123),
        )

        torch.testing.assert_close(time_a, time_b)

    def test_midpoint_integration_advances_constant_field(self) -> None:
        start = torch.zeros(2, 1, 4)

        result = integrate_euler_midpoint_time(lambda values, time: torch.ones_like(values), start, 4)

        torch.testing.assert_close(result, torch.ones_like(start))


if __name__ == "__main__":
    unittest.main()