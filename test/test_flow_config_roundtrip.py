from __future__ import annotations

import unittest

from lpap.flow_training import (
    FlowImageConfig,
    FlowModelConfig,
    FlowOptimizerConfig,
    FlowTimeConfig,
    FlowValidationConfig,
    flow_model_config_from_dict,
    image_config_from_dict,
    optimizer_config_from_dict,
    time_config_from_dict,
    validation_config_from_dict,
)


class FlowConfigRoundTripTest(unittest.TestCase):
    def test_image_config_round_trip(self) -> None:
        config = FlowImageConfig(
            dataset_path="data/images.pt",
            batch_size=8,
            side=32,
            normalize=True,
            shuffle=False,
            num_workers=2,
        )
        self.assertEqual(image_config_from_dict(config.as_dict()), config)

    def test_model_config_round_trip(self) -> None:
        config = FlowModelConfig(
            sequence_length=1024,
            width=64,
            time_dim=32,
            dilation_cycles=2,
            dilations=(1, 2, 4),
            kernel_size=3,
            zero_init_output=True,
        )
        self.assertEqual(flow_model_config_from_dict(config.as_dict()), config)

    def test_time_config_round_trip(self) -> None:
        config = FlowTimeConfig(
            distribution="beta",
            beta_alpha=1.5,
            beta_beta=2.0,
            eps=1.0e-4,
        )
        self.assertEqual(time_config_from_dict(config.as_dict()), config)

    def test_optimizer_config_round_trip(self) -> None:
        config = FlowOptimizerConfig(learning_rate=1.0e-3, max_grad_norm=1.0)
        self.assertEqual(optimizer_config_from_dict(config.as_dict()), config)

        unclipped = FlowOptimizerConfig(learning_rate=5.0e-4, max_grad_norm=None)
        self.assertEqual(optimizer_config_from_dict(unclipped.as_dict()), unclipped)

    def test_validation_config_round_trip(self) -> None:
        config = FlowValidationConfig(
            enabled=True,
            every=10,
            batch_size=4,
            seed=7,
            validate_at_end=True,
            euler_steps=(1, 2, 4),
        )
        self.assertEqual(validation_config_from_dict(config.as_dict()), config)


if __name__ == "__main__":
    unittest.main()
