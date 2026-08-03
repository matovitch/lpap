from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from lpap.checkpoints import save_training_checkpoint
from lpap.decoder import LPAPDecoderTransformer
from lpap.energy_bank import EnergyBankConfig
from lpap.flow import DilatedConvFlow1d
from lpap.image_autoencoder_training import (
    ImageAutoencoderIntegrationConfig,
    ImageAutoencoderLossConfig,
    ImageAutoencoderLpapPairConfig,
    ImageAutoencoderMetrics,
    ImageAutoencoderRunConfig,
    ImageAutoencoderSourceConfig,
    ImageAutoencoderTrainingConfig,
    ImageAutoencoderTrainingSession,
    average_image_autoencoder_metrics,
    collect_image_autoencoder_gallery,
    create_image_autoencoder_training_session,
    evaluate_image_autoencoder_batch,
    evaluate_image_autoencoder_validation,
    image_autoencoder_source_config_from_dict,
    image_autoencoder_training_config_from_dict,
    iter_image_autoencoder_training,
    train_image_autoencoder_step,
)
from lpap.flow_training import (
    FlowImageConfig,
    FlowModelConfig,
    FlowOptimizerConfig,
    FlowValidationConfig,
)
from lpap.permutation import make_grouped_permutation_indices
from lpap.surrogate import LPAPSurrogateTransformer
from lpap.surrogate_training import LPAPSurrogateDataConfig


def _save_tiny_teacher_pair(
    checkpoint_dir: Path,
    *,
    surrogate_name: str,
    decoder_name: str,
    bucket_count: int,
    permutation_seed: int,
    value_count: int = 16,
) -> None:
    probe_count = value_count // bucket_count
    surrogate = LPAPSurrogateTransformer(
        value_count=value_count,
        probe_count=probe_count,
        k_max=2,
        hidden_dim=16,
        layer_count=1,
        head_count=4,
    )
    permutation = make_grouped_permutation_indices(
        value_count=value_count,
        bucket_count=bucket_count,
        seed=permutation_seed,
        device=torch.device("cpu"),
    )
    save_training_checkpoint(
        checkpoint_dir / surrogate_name,
        model=surrogate,
        step=1,
        training_state={
            "run_config": {
                "data": LPAPSurrogateDataConfig(
                    batch_size=2,
                    bucket_count=bucket_count,
                    probe_count=probe_count,
                    energy_bank=EnergyBankConfig(path="data/encoded_energies_ae_best.pt"),
                ).as_dict()
            },
            "model_config": {
                "value_count": value_count,
                "bucket_count": bucket_count,
                "probe_count": probe_count,
                "k_max": 2,
                "hidden_dim": 16,
                "layer_count": 1,
                "head_count": 4,
                "permutation_seed": permutation_seed,
            },
            "permutation": permutation,
        },
    )
    decoder = LPAPDecoderTransformer(
        value_count=value_count,
        frontend_initial_temperature=0.5,
        hidden_dim=16,
        layer_count=1,
        head_count=4,
    )
    save_training_checkpoint(
        checkpoint_dir / decoder_name,
        model=decoder,
        step=1,
        training_state={
            "model_config": {
                "value_count": value_count,
                "bucket_count": bucket_count,
                "probe_count": probe_count,
                "surrogate": {},
                "frontend_initial_temperature": 0.5,
                "hidden_dim": 16,
                "layer_count": 1,
                "head_count": 4,
                "permutation_seed": permutation_seed,
            }
        },
    )


def _build_tiny_autoencoder_session(
    root: Path,
    *,
    lpap_pairs: tuple[ImageAutoencoderLpapPairConfig, ...] | None = None,
    validation: FlowValidationConfig | None = None,
) -> ImageAutoencoderTrainingSession:
    checkpoint_dir = root / "checkpoints"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    torch.save(
        {
            "images": torch.arange(8 * 1 * 4 * 4, dtype=torch.uint8).reshape(
                8, 1, 4, 4
            ),
            "names": [str(index) for index in range(8)],
        },
        data_dir / "images.pt",
    )
    pairs = lpap_pairs or (
        ImageAutoencoderLpapPairConfig(
            surrogate_checkpoint_name="surrogate.pt",
            decoder_checkpoint_name="decoder.pt",
            name="c4",
        ),
    )
    _save_tiny_teacher_pair(
        checkpoint_dir,
        surrogate_name="surrogate.pt",
        decoder_name="decoder.pt",
        bucket_count=4,
        permutation_seed=123,
    )
    if any(
        pair.surrogate_checkpoint_name == "surrogate_b.pt"
        or pair.decoder_checkpoint_name == "decoder_b.pt"
        for pair in pairs
    ):
        _save_tiny_teacher_pair(
            checkpoint_dir,
            surrogate_name="surrogate_b.pt",
            decoder_name="decoder_b.pt",
            bucket_count=8,
            permutation_seed=456,
        )
    flow_kwargs = {
        "sequence_length": 16,
        "width": 8,
        "time_dim": 8,
        "dilation_cycles": 1,
        "dilations": (1, 2),
    }
    flow = DilatedConvFlow1d(**flow_kwargs)
    save_training_checkpoint(
        checkpoint_dir / "image_energy_flow.pt",
        model=flow,
        step=1,
        training_state={"model_config": {"sequence_length": 16}},
    )
    flow_config = FlowModelConfig(**flow_kwargs)
    config = ImageAutoencoderTrainingConfig(
        image=FlowImageConfig(
            dataset_path="data/images.pt",
            batch_size=2,
            side=4,
            normalize=True,
            shuffle=False,
        ),
        source=ImageAutoencoderSourceConfig(
            lpap_pairs=pairs,
            flow_checkpoint_name="image_energy_flow.pt",
            train_image_to_energy_flow=True,
            train_surrogate=True,
            train_decoder=True,
            train_energy_to_image_flow=True,
        ),
        image_to_energy_flow=flow_config,
        energy_to_image_flow=flow_config,
        integration=ImageAutoencoderIntegrationConfig(
            image_to_energy_steps=1,
            energy_to_image_steps=1,
        ),
        loss=ImageAutoencoderLossConfig(
            image_l2_weight=1.0,
            energy_l1_weight=0.25,
            surrogate_teacher_weight=0.1,
        ),
        optimizer=FlowOptimizerConfig(
            learning_rate=1.0e-4,
            max_grad_norm=1.0,
        ),
        validation=validation
        or FlowValidationConfig(
            every=1,
            batch_size=2,
            euler_steps=(1,),
        ),
        run=ImageAutoencoderRunConfig(
            steps=1,
            display_every=1,
            run_id="tiny-image-autoencoder",
            checkpoint_name="image_autoencoder.pt",
            log_name="image_autoencoder.sqlite",
        ),
    )
    return create_image_autoencoder_training_session(
        project_root=root, config=config, device="cpu"
    )


class ImageAutoencoderTrainingTest(unittest.TestCase):
    def test_source_dict_normalizes_to_one_pair(self) -> None:
        source = image_autoencoder_source_config_from_dict(
            {
                "surrogate_checkpoint_name": "surrogate_synthetic.pt",
                "decoder_checkpoint_name": "decoder_synthetic.pt",
                "flow_checkpoint_name": "image_energy_flow.pt",
                "load_best": True,
                "require_checkpoints": True,
                "train_image_to_energy_flow": True,
                "train_surrogate": True,
                "train_decoder": True,
                "train_energy_to_image_flow": True,
            }
        )
        self.assertEqual(len(source.lpap_pairs), 1)
        self.assertEqual(
            source.surrogate_checkpoint_name, "surrogate_synthetic.pt"
        )

    def test_lpap_pairs_source_dict_roundtrip(self) -> None:
        source = image_autoencoder_source_config_from_dict(
            {
                "lpap_pairs": [
                    {
                        "surrogate_checkpoint_name": "surrogate_a.pt",
                        "decoder_checkpoint_name": "decoder_a.pt",
                        "name": "c4",
                    },
                    {
                        "surrogate_checkpoint_name": "surrogate_b.pt",
                        "decoder_checkpoint_name": "decoder_b.pt",
                        "name": "c8",
                    },
                ],
                "flow_checkpoint_name": "image_energy_flow.pt",
                "load_best": True,
                "require_checkpoints": True,
                "train_image_to_energy_flow": True,
                "train_surrogate": False,
                "train_decoder": False,
                "train_energy_to_image_flow": True,
            }
        )
        self.assertEqual(len(source.lpap_pairs), 2)
        restored = image_autoencoder_source_config_from_dict(source.as_dict())
        self.assertEqual(
            [pair.name for pair in restored.lpap_pairs],
            ["c4", "c8"],
        )

    def test_session_trains_and_logs_total_autoencoder_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = _build_tiny_autoencoder_session(root)

            results = list(iter_image_autoencoder_training(session))
            gallery = collect_image_autoencoder_gallery(session, sample_count=1)

            self.assertEqual(len(results), 1)
            self.assertTrue(session.checkpoint_path.exists())
            self.assertTrue(session.log_path.exists())
            self.assertIn("image_reconstruction_l2", results[-1].metrics)
            self.assertIn("energy_reconstruction_l1", results[-1].metrics)
            self.assertIn("surrogate_teacher_ce", results[-1].metrics)
            self.assertIn("image_reconstruction_l2/c4", results[-1].metrics)
            self.assertIn("signed_mass_balance", results[-1].metrics)
            self.assertIn("signed_mass_imbalance", results[-1].metrics)
            self.assertIn("signed_mass_gap", results[-1].metrics)
            self.assertIn("signed_mass_floor", results[-1].metrics)
            self.assertIn("encoded_positive_mass", results[-1].metrics)
            self.assertIn("validation_image_reconstruction_l2", results[-1].metrics)
            self.assertEqual(len(gallery), 1)
            self.assertEqual(gallery[0].image.shape, (4, 4))
            self.assertEqual(gallery[0].encoded_energy.shape, (4, 4))
            self.assertEqual(gallery[0].reconstructed_image.shape, (4, 4))
            self.assertEqual(gallery[0].pairs[0].decoded_energy.shape, (4, 4))
            self.assertEqual(len(gallery[0].pairs), 1)
            self.assertEqual(gallery[0].pairs[0].name, "c4")

    def test_two_pair_session_means_pair_losses_and_backprops_both_branches(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = _build_tiny_autoencoder_session(
                root,
                lpap_pairs=(
                    ImageAutoencoderLpapPairConfig(
                        surrogate_checkpoint_name="surrogate.pt",
                        decoder_checkpoint_name="decoder.pt",
                        name="c4",
                    ),
                    ImageAutoencoderLpapPairConfig(
                        surrogate_checkpoint_name="surrogate_b.pt",
                        decoder_checkpoint_name="decoder_b.pt",
                        name="c8",
                    ),
                ),
            )
            self.assertEqual(len(session.lpap_pairs), 2)
            self.assertEqual(len(session.model.surrogates), 2)
            self.assertEqual(len(session.model.decoders), 2)

            results = list(iter_image_autoencoder_training(session))
            metrics = results[-1].metrics
            self.assertIn("image_reconstruction_l2/c4", metrics)
            self.assertIn("image_reconstruction_l2/c8", metrics)
            self.assertIn("energy_reconstruction_l1/c4", metrics)
            self.assertIn("energy_reconstruction_l1/c8", metrics)
            self.assertIn("surrogate_teacher_ce/c4", metrics)
            self.assertIn("surrogate_teacher_ce/c8", metrics)
            self.assertAlmostEqual(
                metrics["image_reconstruction_l2"],
                0.5
                * (
                    metrics["image_reconstruction_l2/c4"]
                    + metrics["image_reconstruction_l2/c8"]
                ),
                places=5,
            )
            gallery = collect_image_autoencoder_gallery(session, sample_count=1)
            self.assertEqual([pair.name for pair in gallery[0].pairs], ["c4", "c8"])

            images = torch.arange(2 * 1 * 4 * 4, dtype=torch.uint8).reshape(2, 1, 4, 4)
            train_image_autoencoder_step(session=session, images=images)
            for index, (surrogate, decoder) in enumerate(
                zip(session.model.surrogates, session.model.decoders, strict=True)
            ):
                for name, module in (
                    (f"surrogate{index}", surrogate),
                    (f"decoder{index}", decoder),
                ):
                    grads = [
                        parameter.grad
                        for parameter in module.parameters()
                        if parameter.requires_grad
                    ]
                    self.assertTrue(
                        any(grad is not None and torch.any(grad != 0) for grad in grads),
                        f"{name} received no non-zero gradient",
                    )

    def test_training_step_propagates_gradients_through_discrete_bottleneck(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = _build_tiny_autoencoder_session(root)
            images = torch.arange(2 * 1 * 4 * 4, dtype=torch.uint8).reshape(2, 1, 4, 4)

            train_image_autoencoder_step(session=session, images=images)

            submodules = {
                "image_to_energy_flow": session.model.image_to_energy_flow,
                "surrogate": session.model.surrogate,
                "decoder": session.model.decoder,
                "energy_to_image_flow": session.model.energy_to_image_flow,
            }
            for name, module in submodules.items():
                grads = [
                    parameter.grad
                    for parameter in module.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(
                    any(grad is not None for grad in grads),
                    f"{name} received no gradient",
                )
                self.assertTrue(
                    any(grad is not None and torch.any(grad != 0) for grad in grads),
                    f"{name} received only zero gradients",
                )

    def test_from_dict_accepts_flat_source(self) -> None:
        config = image_autoencoder_training_config_from_dict(
            {
                "image": {
                    "dataset_path": "data/images.pt",
                    "batch_size": 2,
                    "side": 4,
                    "normalize": True,
                    "shuffle": False,
                    "num_workers": 0,
                },
                "source": {
                    "surrogate_checkpoint_name": "surrogate.pt",
                    "decoder_checkpoint_name": "decoder.pt",
                    "flow_checkpoint_name": "image_energy_flow.pt",
                    "load_best": True,
                    "require_checkpoints": True,
                    "train_image_to_energy_flow": True,
                    "train_surrogate": True,
                    "train_decoder": True,
                    "train_energy_to_image_flow": True,
                },
                "image_to_energy_flow": {
                    "sequence_length": 16,
                    "width": 8,
                    "time_dim": 8,
                    "dilation_cycles": 1,
                    "dilations": [1, 2],
                    "kernel_size": 3,
                    "zero_init_output": True,
                },
                "energy_to_image_flow": {
                    "sequence_length": 16,
                    "width": 8,
                    "time_dim": 8,
                    "dilation_cycles": 1,
                    "dilations": [1, 2],
                    "kernel_size": 3,
                    "zero_init_output": True,
                },
                "integration": {
                    "image_to_energy_steps": 1,
                    "energy_to_image_steps": 1,
                },
                "loss": {
                    "image_l2_weight": 1.0,
                    "energy_l1_weight": 0.25,
                    "surrogate_teacher_weight": 0.1,
                    "signed_mass_balance_weight": 0.02,
                    "signed_mass_floor_tau": 0.01,
                    "signed_mass_floor_coef": 1.0,
                    "detach_energy_target": False,
                },
                "optimizer": {"learning_rate": 1.0e-4, "max_grad_norm": 1.0},
                "validation": {
                    "enabled": True,
                    "every": 1,
                    "batch_size": 2,
                    "seed": 1,
                    "validate_at_end": True,
                    "euler_steps": [1],
                },
                "run": {
                    "run_training": True,
                    "resume_from_checkpoint": False,
                    "steps": 1,
                    "seed": 1,
                    "display_every": 1,
                    "log_every": 1,
                    "run_id": "flat-source",
                    "checkpoint_name": "image_autoencoder.pt",
                    "log_name": "image_autoencoder.sqlite",
                },
            }
        )
        self.assertEqual(len(config.source.lpap_pairs), 1)
        self.assertEqual(config.validation.num_batches, 1)

    def test_average_image_autoencoder_metrics_means_scalars_and_pair_keys(
        self,
    ) -> None:
        left = ImageAutoencoderMetrics(
            loss=1.0,
            image_reconstruction_l2=2.0,
            energy_reconstruction_l1=3.0,
            surrogate_teacher_ce=4.0,
            surrogate_weighted_accuracy=0.2,
            signed_mass_balance=0.1,
            signed_mass_imbalance=0.2,
            signed_mass_gap=0.3,
            signed_mass_floor=0.4,
            encoded_positive_mass=0.5,
            encoded_negative_mass=0.6,
            encoded_energy_rms=0.7,
            decoded_energy_rms=0.8,
            reconstructed_image_rms=0.9,
            image_rms=1.1,
            pair_metrics={"image_reconstruction_l2/c4": 2.0},
        )
        right = ImageAutoencoderMetrics(
            loss=3.0,
            image_reconstruction_l2=4.0,
            energy_reconstruction_l1=5.0,
            surrogate_teacher_ce=6.0,
            surrogate_weighted_accuracy=0.4,
            signed_mass_balance=0.3,
            signed_mass_imbalance=0.4,
            signed_mass_gap=0.5,
            signed_mass_floor=0.6,
            encoded_positive_mass=0.7,
            encoded_negative_mass=0.8,
            encoded_energy_rms=0.9,
            decoded_energy_rms=1.0,
            reconstructed_image_rms=1.1,
            image_rms=1.3,
            pair_metrics={"image_reconstruction_l2/c4": 6.0},
        )
        mean = average_image_autoencoder_metrics([left, right])
        self.assertAlmostEqual(mean.loss, 2.0)
        self.assertAlmostEqual(mean.image_reconstruction_l2, 3.0)
        self.assertAlmostEqual(mean.pair_metrics["image_reconstruction_l2/c4"], 4.0)

    def test_validation_averages_configured_num_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = _build_tiny_autoencoder_session(
                root,
                validation=FlowValidationConfig(
                    every=1,
                    batch_size=2,
                    num_batches=3,
                    euler_steps=(1,),
                ),
            )
            calls: list[int] = []
            real_evaluate = evaluate_image_autoencoder_batch

            def _counting_evaluate(*, session, images):
                calls.append(int(images.shape[0]))
                return real_evaluate(session=session, images=images)

            with mock.patch(
                "lpap.image_autoencoder_training.evaluate_image_autoencoder_batch",
                side_effect=_counting_evaluate,
            ):
                results = list(iter_image_autoencoder_training(session))

            self.assertEqual(len(results), 1)
            self.assertEqual(len(calls), 3)
            self.assertIn("validation_loss", results[-1].metrics)

    def test_evaluate_validation_rejects_non_positive_num_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = _build_tiny_autoencoder_session(root)
            with self.assertRaises(ValueError):
                evaluate_image_autoencoder_validation(
                    session=session,
                    images_iter=iter(()),
                    num_batches=0,
                )


if __name__ == "__main__":
    unittest.main()
