from __future__ import annotations

from pathlib import Path

import torch

from lpap.checkpoints import load_training_checkpoint
from lpap.data import load_image_tensor_dataset
from lpap.decoder_training import (
    collect_lpap_decoder_gallery,
    create_lpap_decoder_training_session,
    lpap_decoder_training_config_from_dict,
)
from lpap.energy_to_image_training import (
    collect_energy_to_image_gallery,
    create_energy_to_image_training_session,
    energy_to_image_training_config_from_dict,
)
from lpap.energy_to_image_reflow_training import (
    collect_energy_to_image_reflow_gallery,
    create_energy_to_image_reflow_training_session,
    energy_to_image_reflow_training_config_from_dict,
)
from lpap.flow import DilatedConvFlow1d
from lpap.image_to_energy_training import (
    collect_image_to_energy_gallery,
    image_to_energy_training_config_from_dict,
)
from lpap.image_autoencoder_training import (
    collect_image_autoencoder_gallery,
    create_image_autoencoder_training_session,
    image_autoencoder_training_config_from_dict,
)
from lpap.training_log import load_run_record
from lpap.training_plots import (
    render_energy_to_image_gallery_html,
    render_energy_to_image_reflow_gallery_html,
    render_image_autoencoder_gallery_html,
    render_image_to_energy_gallery_html,
    render_signed_triplet_gallery_html,
)


def render_decoder_run_gallery(
    *,
    project_root: str | Path,
    log_path: str | Path,
    run_id: str,
    sample_count: int = 3,
) -> str:
    root = Path(project_root)
    record = load_run_record(log_path, run_id=run_id)
    config = lpap_decoder_training_config_from_dict(
        record["config"], resume_from_checkpoint=False
    )
    session = create_lpap_decoder_training_session(project_root=root, config=config)
    checkpoint_path = Path(record["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    payload = load_training_checkpoint(checkpoint_path, map_location=session.device)
    state = payload.get("best_model_state")
    if state is None:
        state = payload["model_state"]
    session.decoder.load_state_dict(state)
    return render_signed_triplet_gallery_html(
        collect_lpap_decoder_gallery(session, sample_count=sample_count)
    )


def render_image_to_energy_run_gallery(
    *,
    project_root: str | Path,
    log_path: str | Path,
    run_id: str,
    sample_count: int = 3,
    integration_steps: tuple[int, ...] = (64, 32, 16, 8, 4),
) -> str:
    root = Path(project_root)
    record = load_run_record(log_path, run_id=run_id)
    config = image_to_energy_training_config_from_dict(
        record["config"], resume_from_checkpoint=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow = DilatedConvFlow1d(**config.flow.as_dict()).to(device)
    checkpoint_path = Path(record["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    payload = load_training_checkpoint(checkpoint_path, map_location=device)
    state = payload.get("best_model_state")
    if state is None:
        state = payload["model_state"]
    flow.load_state_dict(state)

    dataset_path = Path(config.image.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    dataset = load_image_tensor_dataset(dataset_path, normalize=config.image.normalize)
    count = min(sample_count, len(dataset))
    if count <= 0:
        return "<p>No image samples are available.</p>"
    images = torch.stack([dataset[index][0] for index in range(count)])
    return render_image_to_energy_gallery_html(
        collect_image_to_energy_gallery(
            model=flow,
            images=images,
            side=config.image.side,
            steps=integration_steps,
            device=device,
        ),
        steps=integration_steps,
        size=config.image.side,
    )


def render_energy_to_image_run_gallery(
    *,
    project_root: str | Path,
    log_path: str | Path,
    run_id: str,
    sample_count: int = 3,
    integration_steps: tuple[int, ...] = (64, 32, 16, 8, 4),
) -> str:
    root = Path(project_root)
    record = load_run_record(log_path, run_id=run_id)
    config = energy_to_image_training_config_from_dict(
        record["config"], resume_from_checkpoint=False
    )
    session = create_energy_to_image_training_session(project_root=root, config=config)
    checkpoint_path = Path(record["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    payload = load_training_checkpoint(checkpoint_path, map_location=session.device)
    state = payload.get("best_model_state")
    if state is None:
        state = payload["model_state"]
    session.flow.load_state_dict(state)
    return render_energy_to_image_gallery_html(
        collect_energy_to_image_gallery(
            session,
            sample_count=sample_count,
            steps=integration_steps,
        ),
        steps=integration_steps,
        size=config.image.side,
    )


def render_energy_to_image_reflow_run_gallery(
    *,
    project_root: str | Path,
    log_path: str | Path,
    run_id: str,
    sample_count: int = 3,
) -> str:
    root = Path(project_root)
    record = load_run_record(log_path, run_id=run_id)
    config = energy_to_image_reflow_training_config_from_dict(
        record["config"], resume_from_checkpoint=False
    )
    session = create_energy_to_image_reflow_training_session(
        project_root=root, config=config
    )
    checkpoint_path = Path(record["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    payload = load_training_checkpoint(checkpoint_path, map_location=session.device)
    state = payload.get("best_model_state")
    if state is None:
        state = payload["model_state"]
    session.student_flow.load_state_dict(state)
    return render_energy_to_image_reflow_gallery_html(
        collect_energy_to_image_reflow_gallery(
            session,
            sample_count=sample_count,
        ),
        size=config.image.side,
    )


def render_image_autoencoder_run_gallery(
    *,
    project_root: str | Path,
    log_path: str | Path,
    run_id: str,
    sample_count: int = 3,
) -> str:
    root = Path(project_root)
    record = load_run_record(log_path, run_id=run_id)
    config = image_autoencoder_training_config_from_dict(
        record["config"], resume_from_checkpoint=False
    )
    session = create_image_autoencoder_training_session(
        project_root=root, config=config
    )
    checkpoint_path = Path(record["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    payload = load_training_checkpoint(checkpoint_path, map_location=session.device)
    state = payload.get("best_model_state")
    if state is None:
        state = payload["model_state"]
    session.model.load_state_dict(state)
    return render_image_autoencoder_gallery_html(
        collect_image_autoencoder_gallery(
            session,
            sample_count=sample_count,
        ),
        size=config.image.side,
    )
