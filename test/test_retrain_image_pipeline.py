from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from lpap.checkpoints import load_training_checkpoint, save_training_checkpoint
from lpap.flow import DilatedConvFlow1d
from lpap.permutation import LPAP_FOLD_ORDER
from lpap.retrain_image_pipeline import (
    ALL_PIPELINE_STAGES,
    load_retrain_pipeline_config,
    run_image_retrain_pipeline,
)


def _write_tiny_pipeline_workspace(root: Path) -> Path:
    """Create a tiny self-contained workspace (4x4 images, N=16, 2 pairs)."""
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg_dir = root / "configs" / "training"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "images": torch.randint(0, 256, (12, 1, 4, 4), dtype=torch.uint8),
            "names": [f"img_{i}" for i in range(12)],
        },
        root / "data" / "images.pt",
    )
    torch.save(
        {"energies": torch.randn(16, 16, generator=torch.Generator().manual_seed(3))},
        root / "data" / "bank.pt",
    )
    flow = DilatedConvFlow1d(
        sequence_length=16,
        width=8,
        time_dim=8,
        dilation_cycles=1,
        dilations=(1, 2),
        kernel_size=3,
        zero_init_output=True,
    )
    save_training_checkpoint(
        root / "checkpoints" / "image_energy_flow.pt",
        model=flow,
        step=1,
        training_state={},
    )

    for name, c, p, k, seed in (
        ("c4_k2", 4, 4, 2, 11),
        ("c8_k3", 8, 2, 3, 22),
    ):
        (cfg_dir / f"teacher_{name}.toml").write_text(
            f"""[pair]
name = "{name}"
bucket_count = {c}
probe_count = {p}
k_max = {k}
permutation_seed = {seed}

[pair.energy_bank]
path = "data/bank.pt"
energies_key = "energies"

[data]
batch_size = 4

[surrogate]
hidden_dim = 16
layer_count = 1
head_count = 4
learning_rate = 1.0e-3

[surrogate.validation]
enabled = true
every = 1
batch_size = 4
seed = 101
validate_at_end = true

[surrogate.run]
steps = 2
seed = 1
display_every = 1
log_every = 1

[decoder]
frontend_initial_temperature = 0.25
hidden_dim = 16
layer_count = 1
head_count = 4
learning_rate = 1.0e-3
source_ce_weight = 0.1
source_ce_l1_reference = 0.05
source_ce_power = 2.0
load_best = true
require_checkpoint = true

[decoder.validation]
enabled = true
every = 1
batch_size = 4
seed = 202
validate_at_end = true

[decoder.run]
steps = 2
seed = 2
display_every = 1
log_every = 1
""",
            encoding="utf-8",
        )

    (cfg_dir / "image_autoencoder.toml").write_text(
        """[image]
dataset_path = "data/images.pt"
batch_size = 4
side = 4
normalize = true
shuffle = false
num_workers = 0

[source]
flow_checkpoint_name = "image_energy_flow.pt"
load_best = true
require_checkpoints = true
train_image_to_energy_flow = true
train_surrogate = true
train_decoder = true
train_energy_to_image_flow = true
lpap_frontend_mode = "surrogate"
init_checkpoint_name = ""
use_triton = false

[[source.lpap_pairs]]
name = "c4_k2"
surrogate_checkpoint_name = "surrogate_c4_k2.pt"
decoder_checkpoint_name = "decoder_c4_k2.pt"

[[source.lpap_pairs]]
name = "c8_k3"
surrogate_checkpoint_name = "surrogate_c8_k3.pt"
decoder_checkpoint_name = "decoder_c8_k3.pt"

[image_to_energy_flow]
sequence_length = 16
width = 8
time_dim = 8
dilation_cycles = 1
dilations = [1, 2]
kernel_size = 3
zero_init_output = true

[energy_to_image_flow]
sequence_length = 16
width = 8
time_dim = 8
dilation_cycles = 1
dilations = [1, 2]
kernel_size = 3
zero_init_output = true

[integration]
image_to_energy_steps = 1
energy_to_image_steps = 1

[loss]
image_l2_weight = 1.0
energy_l1_weight = 0.05
surrogate_teacher_weight = 0.05
signed_mass_balance_weight = 0.01
signed_mass_floor_tau = 0.1
signed_mass_floor_coef = 1.0
detach_energy_target = false

[optimizer]
learning_rate = 1.0e-3
max_grad_norm = 1.0

[validation]
enabled = true
every = 1
batch_size = 4
num_batches = 1
seed = 303
validate_at_end = true
euler_steps = [1]

[run]
run_training = true
resume_from_checkpoint = false
steps = 2
seed = 3
display_every = 1
log_every = 1
run_id = "ae_smoke"
checkpoint_name = "ae_smoke.pt"
log_name = "ae_smoke.sqlite"
""",
        encoding="utf-8",
    )

    pipeline_path = cfg_dir / "retrain_image_pipeline.toml"
    pipeline_path.write_text(
        """[pipeline]
flow_checkpoint = "image_energy_flow.pt"
energy_bank = "data/bank.pt"
image_dataset = "data/images.pt"
teacher_configs = [
  "configs/training/teacher_c4_k2.toml",
  "configs/training/teacher_c8_k3.toml",
]
autoencoder_config = "configs/training/image_autoencoder.toml"
resume_from_checkpoint = false
ensure_artifacts = false
upload_artifacts_on_checkpoint = false
notify_on_finished = false

[stages]
run_surrogates = true
run_decoders = true
run_autoencoder = true
run_decoder_exact_finetune = true

[decoder_exact_finetune]
steps = 2
learning_rate = 1.0e-3
use_triton = false
image_l2_weight = 1.0
energy_l1_weight = 0.05
validation_every = 1
validation_num_batches = 1
display_every = 1
log_every = 1
run_id = "ae_exact_ft_smoke"
checkpoint_name = "ae_exact_ft_smoke.pt"
log_name = "ae_exact_ft_smoke.sqlite"
comment = "smoke exact fine-tune"
""",
        encoding="utf-8",
    )
    return pipeline_path


class RetrainImagePipelineTest(unittest.TestCase):
    def test_live_pipeline_toml_parses_and_uses_v2_artifact_names(self) -> None:
        from lpap.teacher_config import project_teacher_config
        from lpap.training_notebook import training_config_from_file

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_retrain_pipeline_config(
            repo_root / "configs" / "training" / "retrain_image_pipeline.toml"
        )
        self.assertEqual(cfg.enabled_stages(), ALL_PIPELINE_STAGES)
        self.assertEqual(len(cfg.teacher_configs), 3)
        self.assertEqual(cfg.flow_checkpoint, "image_energy_flow.pt")
        self.assertEqual(
            cfg.energy_bank, "data/encoded_energies_bank_flow_best_4step.pt"
        )

        expected_surrogates: list[str] = []
        expected_decoders: list[str] = []
        for relpath in cfg.teacher_configs:
            surr = project_teacher_config(repo_root / relpath, "surrogate")
            dec = project_teacher_config(repo_root / relpath, "decoder")
            self.assertTrue(surr.run.run_id.endswith("-v2"))
            self.assertTrue(surr.run.checkpoint_name.endswith("-v2.pt"))
            self.assertTrue(surr.run.log_name.endswith("-v2.sqlite"))
            self.assertTrue(dec.run.run_id.endswith("-v2"))
            self.assertTrue(dec.run.checkpoint_name.endswith("-v2.pt"))
            self.assertTrue(dec.run.log_name.endswith("-v2.sqlite"))
            self.assertEqual(dec.teacher.checkpoint_name, surr.run.checkpoint_name)
            expected_surrogates.append(surr.run.checkpoint_name)
            expected_decoders.append(dec.run.checkpoint_name)

        ae_cfg = training_config_from_file(
            repo_root / cfg.autoencoder_config, "image_autoencoder"
        )
        self.assertEqual(ae_cfg.source.flow_checkpoint_name, "image_energy_flow.pt")
        self.assertEqual(ae_cfg.integration.image_to_energy_steps, 4)
        self.assertEqual(ae_cfg.integration.energy_to_image_steps, 4)
        self.assertEqual(ae_cfg.validation.euler_steps, (4,))
        self.assertEqual(
            [pair.surrogate_checkpoint_name for pair in ae_cfg.source.lpap_pairs],
            expected_surrogates,
        )
        self.assertEqual(
            [pair.decoder_checkpoint_name for pair in ae_cfg.source.lpap_pairs],
            expected_decoders,
        )
        self.assertEqual(ae_cfg.run.run_id, "image_autoencoder_tri_lnorm-v2")
        self.assertEqual(
            ae_cfg.run.checkpoint_name, "image_autoencoder_tri_lnorm-v2.pt"
        )
        self.assertEqual(
            ae_cfg.run.log_name, "image_autoencoder_tri_lnorm-v2.sqlite"
        )
        self.assertEqual(
            cfg.decoder_exact_finetune.run_id, "image_autoencoder_tri_exact_ft-v2"
        )
        self.assertEqual(
            cfg.decoder_exact_finetune.checkpoint_name,
            "image_autoencoder_tri_exact_ft-v2.pt",
        )
        self.assertEqual(
            cfg.decoder_exact_finetune.log_name,
            "image_autoencoder_tri_exact_ft-v2.sqlite",
        )
        self.assertEqual(cfg.performance.autocast_dtype, "bf16")
        self.assertTrue(cfg.performance.energy_bank_on_device)

    def test_performance_defaults_keep_fp32_behaviour(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = load_retrain_pipeline_config(
                _write_tiny_pipeline_workspace(Path(temp_dir))
            )
        self.assertEqual(cfg.performance.autocast_dtype, "none")
        self.assertFalse(cfg.performance.allow_tf32)
        self.assertFalse(cfg.performance.energy_bank_on_device)
        self.assertFalse(cfg.performance.compile_flows)

    def test_cpu_smoke_with_bf16_autocast_and_bank_on_device(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline_path = _write_tiny_pipeline_workspace(root)
            with pipeline_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n[performance]\nautocast_dtype = \"bf16\"\n"
                    "energy_bank_on_device = true\n"
                )
            cfg = load_retrain_pipeline_config(pipeline_path)
            self.assertEqual(cfg.performance.autocast_dtype, "bf16")
            results = run_image_retrain_pipeline(
                project_root=root, config=cfg, device="cpu"
            )
            self.assertEqual(len(results), 6)
            for item in results:
                self.assertTrue(Path(item.checkpoint_path).is_file())
                loss = item.last_metrics.get("loss")
                self.assertIsNotNone(loss)
                self.assertTrue(loss == loss, f"NaN loss in {item.name}")

    def test_end_to_end_cpu_smoke_runs_all_four_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline_path = _write_tiny_pipeline_workspace(root)
            cfg = load_retrain_pipeline_config(pipeline_path)

            results = run_image_retrain_pipeline(
                project_root=root,
                config=cfg,
                device="cpu",
            )

            # 2 surrogates + 2 decoders + 1 AE + 1 exact fine-tune = 6 stage runs.
            self.assertEqual(len(results), 6)
            self.assertEqual(
                [item.stage for item in results],
                [
                    "surrogates",
                    "surrogates",
                    "decoders",
                    "decoders",
                    "autoencoder",
                    "decoder_exact_finetune",
                ],
            )
            for item in results:
                ckpt = Path(item.checkpoint_path)
                self.assertTrue(ckpt.is_file(), f"missing checkpoint: {ckpt}")
                payload = load_training_checkpoint(ckpt)
                self.assertEqual(
                    payload["training_state"]["fold_order"], LPAP_FOLD_ORDER
                )

            # Stage 4 (exact LPAP fine-tune) records zero surrogate CE and 1.0 acc.
            ft_result = results[-1]
            self.assertEqual(ft_result.last_metrics["surrogate_teacher_ce"], 0.0)
            self.assertEqual(ft_result.last_metrics["weighted_accuracy"], 1.0)

    def test_pushover_notification_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline_path = _write_tiny_pipeline_workspace(root)
            cfg = load_retrain_pipeline_config(pipeline_path)

            with mock.patch(
                "lpap.retrain_image_pipeline.send_pushover",
                autospec=True,
                return_value={"status": 1},
            ) as mock_push:
                results = run_image_retrain_pipeline(
                    project_root=root,
                    config=cfg,
                    stages=("surrogates",),
                    device="cpu",
                    notify_on_finished=True,
                    step_overrides={"surrogates": 1},
                )

            self.assertEqual(len(results), 2)
            mock_push.assert_called_once()
            args, kwargs = mock_push.call_args
            self.assertEqual(len(args), 1)
            self.assertIn("surrogates/surrogate_c4_k2", args[0])
            self.assertIn("surrogates/surrogate_c8_k3", args[0])
            self.assertEqual(
                kwargs.get("title"), "LPAP image retrain pipeline finished"
            )

    def test_pushover_notification_on_failure_and_reraises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline_path = _write_tiny_pipeline_workspace(root)
            cfg = load_retrain_pipeline_config(pipeline_path)

            with (
                mock.patch(
                    "lpap.retrain_image_pipeline.send_pushover",
                    autospec=True,
                    return_value={"status": 1},
                ) as mock_push,
                mock.patch(
                    "lpap.retrain_image_pipeline.iter_lpap_decoder_training",
                    side_effect=RuntimeError("boom in decoder"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom in decoder"):
                    run_image_retrain_pipeline(
                        project_root=root,
                        config=cfg,
                        stages=("surrogates", "decoders"),
                        device="cpu",
                        notify_on_finished=True,
                        step_overrides={"surrogates": 1, "decoders": 1},
                    )

            mock_push.assert_called_once()
            args, kwargs = mock_push.call_args
            self.assertEqual(len(args), 1)
            self.assertIn("stage=decoders", args[0])
            self.assertIn("RuntimeError: boom in decoder", args[0])
            self.assertIn("done surrogates/surrogate_c4_k2", args[0])
            self.assertEqual(
                kwargs.get("title"), "LPAP image retrain pipeline failed"
            )

    def test_pushover_failure_error_does_not_mask_original_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline_path = _write_tiny_pipeline_workspace(root)
            cfg = load_retrain_pipeline_config(pipeline_path)

            with (
                mock.patch(
                    "lpap.retrain_image_pipeline.send_pushover",
                    autospec=True,
                    side_effect=RuntimeError("notify failed"),
                ),
                mock.patch(
                    "lpap.retrain_image_pipeline.iter_lpap_surrogate_training",
                    side_effect=ValueError("original failure"),
                ),
            ):
                with self.assertWarns(UserWarning):
                    with self.assertRaisesRegex(ValueError, "original failure"):
                        run_image_retrain_pipeline(
                            project_root=root,
                            config=cfg,
                            stages=("surrogates",),
                            device="cpu",
                            notify_on_finished=True,
                            step_overrides={"surrogates": 1},
                        )


if __name__ == "__main__":
    unittest.main()
