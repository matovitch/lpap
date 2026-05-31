"""Public API for the LPAP package.

Core modules (operators, models, data, training scaffolding, checkpoints,
training log) are imported eagerly. Per-model training stacks, notebook
helpers, and rendering utilities are loaded lazily on first access via
PEP 562 ``__getattr__`` so ``import lpap`` stays cheap.
"""

from __future__ import annotations

import importlib
from typing import Any

from lpap.checkpoints import (
    CheckpointInfo,
    load_training_checkpoint,
    metric_improved,
    save_training_checkpoint,
)
from lpap.data import (
    ImageTensorDataset,
    SyntheticHarmonicConfig,
    SyntheticHarmonicDataset,
    image_dataloader,
    load_image_tensor_dataset,
    sample_synthetic_harmonic_batch,
    synthetic_harmonic_dataloader,
)
from lpap.decoder import (
    LPAPDecoderBatch,
    LPAPDecoderMetrics,
    LPAPDecoderTransformer,
    decoder_dibs_from_source_logits,
    evaluate_lpap_decoder_batch,
    lpap_decoder_loss,
    prepare_lpap_decoder_batch,
    reconstruct_lpap_bucket_values,
    reconstruct_lpap_decoder_values,
    train_lpap_decoder_step,
)
from lpap.flow import (
    DilatedConvFlow1d,
    DilatedResidualBlock1d,
    FlowMatchingMetrics,
    SinusoidalTimeEmbedding,
    flow_matching_loss,
    integrate_euler_midpoint_time,
    interpolate_linear,
)
from lpap.hilbert import (
    hilbert_flatten_images,
    hilbert_metadata,
    hilbert_permutation,
    hilbert_unflatten_images,
    inverse_hilbert_permutation,
    inverse_permutation,
)
from lpap.ops import lpap_torch
from lpap.permutation import (
    apply_grouped_permutation,
    fold_grouped_permutation_tokens,
    invert_permutation_indices,
    make_grouped_permutation_indices,
    reverse_grouped_permutation,
    unfold_grouped_permutation_tokens,
)
from lpap.surrogate import (
    LPAPSurrogateMetrics,
    LPAPSurrogateTargets,
    LPAPSurrogateTransformer,
    circular_previous_attention_mask,
    evaluate_lpap_surrogate_batch,
    lpap_surrogate_loss,
    lpap_surrogate_targets,
    prepare_lpap_surrogate_batch,
    train_lpap_surrogate,
    train_lpap_surrogate_step,
)
from lpap.training import (
    TrainingResumeInfo,
    TrainingRun,
    TrainingRunConfig,
    TrainingStepResult,
)
from lpap.training_log import (
    finish_run_attempt,
    initialize_training_log,
    list_training_runs,
    load_best_metric_row,
    load_metric_history,
    load_recent_metrics,
    load_run_record,
    log_step_metrics,
    make_run_display_name,
    make_run_instance_id,
    mark_run_status,
    start_run_attempt,
    upsert_run,
)
from lpap.transformer import RotarySelfAttention, TransformerBlock, apply_rope
from lpap.triton_ops import lpap_triton


# ---------------------------------------------------------------------------
# Lazy submodule loading (PEP 562).
#
# Heavy per-model training stacks, notebook helpers, and rendering utilities
# are imported on first attribute access. The mapping below stays sorted by
# submodule for legibility; adding a new lazy export is a one-line change.
# ---------------------------------------------------------------------------

_LAZY_MODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "decoder_training": (
        "LPAPDecoderGalleryItem",
        "LPAPDecoderModelConfig",
        "LPAPDecoderRegularizationConfig",
        "LPAPDecoderRunConfig",
        "LPAPDecoderTeacherConfig",
        "LPAPDecoderTrainingConfig",
        "LPAPDecoderTrainingSession",
        "collect_lpap_decoder_gallery",
        "create_lpap_decoder_training_session",
        "iter_lpap_decoder_training",
        "lpap_decoder_training_config_from_dict",
        "rerun_lpap_decoder_training_config_from_log",
        "should_validate_lpap_decoder",
        "validate_lpap_decoder",
    ),
    "energy_to_image_reflow_training": (
        "EnergyToImageReflowConfig",
        "EnergyToImageReflowFlowConfig",
        "EnergyToImageReflowGalleryItem",
        "EnergyToImageReflowImageConfig",
        "EnergyToImageReflowMetrics",
        "EnergyToImageReflowOptimizerConfig",
        "EnergyToImageReflowRunConfig",
        "EnergyToImageReflowTeacherConfig",
        "EnergyToImageReflowTrainingConfig",
        "EnergyToImageReflowTrainingSession",
        "EnergyToImageReflowValidationConfig",
        "collect_energy_to_image_reflow_gallery",
        "create_energy_to_image_reflow_training_session",
        "energy_to_image_reflow_training_config_from_dict",
        "evaluate_energy_to_image_reflow_batch",
        "iter_energy_to_image_reflow_training",
        "rerun_energy_to_image_reflow_training_config_from_log",
        "should_validate_energy_to_image_reflow",
        "train_energy_to_image_reflow_step",
    ),
    "energy_to_image_training": (
        "EnergyToImageGalleryItem",
        "EnergyToImageRunConfig",
        "EnergyToImageSourceConfig",
        "EnergyToImageTrainingConfig",
        "EnergyToImageTrainingSession",
        "collect_energy_to_image_gallery",
        "create_energy_to_image_training_session",
        "energy_to_image_training_config_from_dict",
        "evaluate_energy_to_image_batch",
        "iter_energy_to_image_training",
        "rerun_energy_to_image_training_config_from_log",
        "should_validate_energy_to_image",
        "train_energy_to_image_step",
    ),
    "flow_training": (
        "FlowImageConfig",
        "FlowModelConfig",
        "FlowOptimizerConfig",
        "FlowRunParams",
        "FlowSessionCore",
        "FlowTimeConfig",
        "FlowValidationConfig",
        "create_flow_session_core",
        "cycle_image_batches",
        "evaluate_flow_matching_batch",
        "flow_metrics_dict",
        "flow_model_metadata",
        "flow_run_params_from_config",
        "integrate_flow_images",
        "integration_diagnostics",
        "load_flow_checkpoint_state",
        "prepare_image_sequence",
        "sample_flow_time",
        "sample_harmonic_values",
        "should_validate_flow",
        "train_flow_matching_step",
        "validate_image_flow_shape",
    ),
    "image_autoencoder_training": (
        "ImageAutoencoderFlowConfig",
        "ImageAutoencoderForward",
        "ImageAutoencoderGalleryItem",
        "ImageAutoencoderImageConfig",
        "ImageAutoencoderIntegrationConfig",
        "ImageAutoencoderLossConfig",
        "ImageAutoencoderMetrics",
        "ImageAutoencoderModel",
        "ImageAutoencoderOptimizerConfig",
        "ImageAutoencoderRunConfig",
        "ImageAutoencoderSourceConfig",
        "ImageAutoencoderTrainingConfig",
        "ImageAutoencoderTrainingSession",
        "ImageAutoencoderValidationConfig",
        "collect_image_autoencoder_gallery",
        "create_image_autoencoder_training_session",
        "evaluate_image_autoencoder_batch",
        "image_autoencoder_training_config_from_dict",
        "iter_image_autoencoder_training",
        "rerun_image_autoencoder_training_config_from_log",
        "should_validate_image_autoencoder",
        "train_image_autoencoder_step",
    ),
    "image_to_energy_training": (
        "ImageToEnergyFlowConfig",
        "ImageToEnergyGalleryItem",
        "ImageToEnergyImageConfig",
        "ImageToEnergyOptimizerConfig",
        "ImageToEnergyRunConfig",
        "ImageToEnergyTargetConfig",
        "ImageToEnergyTimeConfig",
        "ImageToEnergyTrainingConfig",
        "ImageToEnergyTrainingSession",
        "ImageToEnergyValidationConfig",
        "collect_image_to_energy_gallery",
        "create_image_to_energy_training_session",
        "evaluate_image_to_energy_batch",
        "image_to_energy_training_config_from_dict",
        "iter_image_to_energy_training",
        "rerun_image_to_energy_training_config_from_log",
        "sample_image_to_energy_time",
        "should_validate_image_to_energy",
        "train_image_to_energy_step",
    ),
    "surrogate_training": (
        "LPAPSurrogateDataConfig",
        "LPAPSurrogateModelConfig",
        "LPAPSurrogateOptimizerConfig",
        "LPAPSurrogateRunConfig",
        "LPAPSurrogateTrainingConfig",
        "LPAPSurrogateTrainingSession",
        "LPAPSurrogateValidationConfig",
        "create_lpap_surrogate_training_session",
        "iter_lpap_surrogate_training",
        "lpap_surrogate_training_config_from_dict",
        "rerun_lpap_surrogate_training_config_from_log",
        "should_validate_lpap_surrogate",
        "validate_lpap_surrogate",
    ),
    "training_notebook": (
        "create_training_session",
        "default_decoder_training_config",
        "default_energy_to_image_reflow_training_config",
        "default_energy_to_image_training_config",
        "default_image_autoencoder_training_config",
        "default_image_to_energy_training_config",
        "default_surrogate_training_config",
        "default_training_config",
        "iter_training",
        "recent_training_runs",
        "render_recent_runs_table",
        "restore_training_config_from_log",
        "training_config_from_file",
        "training_config_from_log",
        "training_config_from_project_file",
        "training_config_path",
        "training_config_to_toml",
        "training_log_path",
        "validation_regularizer_metric_names",
        "write_training_config_file",
    ),
    "training_plots": (
        "render_energy_to_image_gallery_html",
        "render_energy_to_image_reflow_gallery_html",
        "render_image_autoencoder_gallery_html",
        "render_image_to_energy_gallery_html",
        "render_loss_history_svg",
        "render_signed_triplet_gallery_html",
    ),
    "visualization_notebook": (
        "render_decoder_run_gallery",
        "render_energy_to_image_reflow_run_gallery",
        "render_energy_to_image_run_gallery",
        "render_image_autoencoder_run_gallery",
        "render_image_to_energy_run_gallery",
    ),
}


_LAZY_NAME_TO_MODULE: dict[str, str] = {
    name: module for module, names in _LAZY_MODULE_EXPORTS.items() for name in names
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module 'lpap' has no attribute {name!r}")
    module = importlib.import_module(f"lpap.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_NAME_TO_MODULE))


__all__ = sorted(
    {
        # Eagerly imported names
        "CheckpointInfo",
        "DilatedConvFlow1d",
        "DilatedResidualBlock1d",
        "FlowMatchingMetrics",
        "ImageTensorDataset",
        "LPAPDecoderBatch",
        "LPAPDecoderMetrics",
        "LPAPDecoderTransformer",
        "LPAPSurrogateMetrics",
        "LPAPSurrogateTargets",
        "LPAPSurrogateTransformer",
        "RotarySelfAttention",
        "SinusoidalTimeEmbedding",
        "SyntheticHarmonicConfig",
        "SyntheticHarmonicDataset",
        "TrainingResumeInfo",
        "TrainingRun",
        "TrainingRunConfig",
        "TrainingStepResult",
        "TransformerBlock",
        "apply_grouped_permutation",
        "apply_rope",
        "circular_previous_attention_mask",
        "decoder_dibs_from_source_logits",
        "evaluate_lpap_decoder_batch",
        "evaluate_lpap_surrogate_batch",
        "finish_run_attempt",
        "flow_matching_loss",
        "fold_grouped_permutation_tokens",
        "hilbert_flatten_images",
        "hilbert_metadata",
        "hilbert_permutation",
        "hilbert_unflatten_images",
        "image_dataloader",
        "initialize_training_log",
        "integrate_euler_midpoint_time",
        "interpolate_linear",
        "inverse_hilbert_permutation",
        "inverse_permutation",
        "invert_permutation_indices",
        "list_training_runs",
        "load_best_metric_row",
        "load_image_tensor_dataset",
        "load_metric_history",
        "load_recent_metrics",
        "load_run_record",
        "load_training_checkpoint",
        "log_step_metrics",
        "lpap_decoder_loss",
        "lpap_surrogate_loss",
        "lpap_surrogate_targets",
        "lpap_torch",
        "lpap_triton",
        "make_grouped_permutation_indices",
        "make_run_display_name",
        "make_run_instance_id",
        "mark_run_status",
        "metric_improved",
        "prepare_lpap_decoder_batch",
        "prepare_lpap_surrogate_batch",
        "reconstruct_lpap_bucket_values",
        "reconstruct_lpap_decoder_values",
        "reverse_grouped_permutation",
        "sample_synthetic_harmonic_batch",
        "save_training_checkpoint",
        "start_run_attempt",
        "synthetic_harmonic_dataloader",
        "train_lpap_decoder_step",
        "train_lpap_surrogate",
        "train_lpap_surrogate_step",
        "unfold_grouped_permutation_tokens",
        "upsert_run",
    }
    | set(_LAZY_NAME_TO_MODULE)
)
