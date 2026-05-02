"""SurgWMBench train/eval adapter for the DIAMOND baseline."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from diamond_surgwmbench.data import SurgWMBenchClipDataset, collate_dense_variable_length, collate_sparse_anchors
from diamond_surgwmbench.evaluation import metrics as traj_metrics

BASELINE_KEY = "diamond"
MODEL_NAME = "SurgWMBenchDiamondDiffusion"


@dataclass
class AdapterConfig:
    target: str
    interpolation_method: str
    image_size: int = 64
    latent_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    diffusion_steps: int = 16
    batch_size: int = 4
    learning_rate: float = 1e-3
    epochs: int = 10
    clean_weight: float = 1.0
    noise_weight: float = 1.0
    recon_weight: float = 0.05
    num_workers: int = 0
    max_clips: int | None = None
    max_frames: int | None = None
    seed: int = 42


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def target_metadata(target: str) -> dict[str, Any]:
    if target == "sparse_20_anchor":
        return {"frame_sampling": "sparse_anchors", "use_dense_pseudo": False, "primary_target": "sparse_human_anchors", "dense_target": None}
    if target == "dense_pseudo":
        return {"frame_sampling": "dense", "use_dense_pseudo": True, "primary_target": "sparse_human_anchors", "dense_target": "pseudo_coordinates"}
    raise ValueError(f"Unsupported target: {target}")


def _collate_for_target(target: str):
    return collate_sparse_anchors if target == "sparse_20_anchor" else collate_dense_variable_length


def make_loader(dataset_root: str | Path, manifest: str, config: AdapterConfig, *, shuffle: bool) -> DataLoader:
    meta = target_metadata(config.target)
    dataset = SurgWMBenchClipDataset(
        dataset_root=dataset_root,
        manifest=manifest,
        interpolation_method=config.interpolation_method,
        image_size=config.image_size,
        frame_sampling=meta["frame_sampling"],
        max_frames=config.max_frames,
        use_dense_pseudo=meta["use_dense_pseudo"],
        return_images=True,
        strict=True,
    )
    if config.max_clips is not None:
        dataset = Subset(dataset, list(range(min(int(config.max_clips), len(dataset)))))
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers, collate_fn=_collate_for_target(config.target))


class FrameEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, latent_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(images).flatten(1))


class FrameDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
        recon = self.net(self.fc(z).view(z.shape[0], 128, 4, 4))
        if recon.shape[-2:] != size_hw:
            recon = F.interpolate(recon, size=size_hw, mode="bilinear", align_corners=False)
        return recon


class SurgWMBenchDiamondDiffusion(nn.Module):
    """DIAMOND-style diffusion world model for SurgWMBench trajectories."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 4, diffusion_steps: int = 16) -> None:
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.encoder = FrameEncoder(latent_dim)
        self.decoder = FrameDecoder(latent_dim)
        self.coord_embed = nn.Linear(2, hidden_dim)
        self.step_embed = nn.Embedding(diffusion_steps, hidden_dim)
        self.input_proj = nn.Linear(latent_dim + 1, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.clean_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(inplace=True), nn.Linear(hidden_dim, 2))
        self.noise_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(inplace=True), nn.Linear(hidden_dim, 2))

    def encode_context(self, frames: torch.Tensor, frame_indices: torch.Tensor, num_frames: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        bsz, timesteps, channels, height, width = frames.shape
        flat = frames.reshape(bsz * timesteps, channels, height, width)
        z = self.encoder(flat)
        recon = self.decoder(z, (height, width)).view(bsz, timesteps, channels, height, width)
        z_seq = z.view(bsz, timesteps, -1)
        denom = torch.clamp(num_frames.to(frames.device, dtype=torch.float32) - 1.0, min=1.0).unsqueeze(1)
        time_feature = frame_indices.to(frames.device, dtype=torch.float32).clamp(min=0).unsqueeze(-1) / denom.unsqueeze(-1)
        hidden = self.input_proj(torch.cat([z_seq, time_feature], dim=-1))
        padding_mask = None if mask is None else ~mask.to(device=frames.device, dtype=torch.bool)
        hidden = self.transformer(hidden, src_key_padding_mask=padding_mask)
        clean = torch.sigmoid(self.clean_head(hidden))
        return {"hidden": hidden, "clean_coords": clean, "recon": recon}

    def forward(
        self,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        num_frames: torch.Tensor,
        noisy_coords: torch.Tensor | None = None,
        diffusion_step: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.encode_context(frames, frame_indices, num_frames, mask)
        if noisy_coords is not None and diffusion_step is not None:
            step = self.step_embed(diffusion_step).unsqueeze(1)
            hidden = out["hidden"] + self.coord_embed(noisy_coords) + step
            out["pred_noise"] = self.noise_head(hidden)
        return out


def batch_mask(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    mask = batch.get("frame_mask")
    if mask is None:
        mask = batch.get("human_anchor_mask")
    if mask is None:
        raise KeyError("Batch does not contain a frame mask")
    return mask.to(device=device, dtype=torch.bool)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    weights = batch.get("label_weight")
    if weights is None:
        weights = torch.ones_like(mask, dtype=torch.float32)
    weights = weights.to(device=pred.device, dtype=torch.float32) * mask.to(dtype=torch.float32)
    loss = (pred - target).square().sum(dim=-1) * weights
    return loss.sum() / weights.sum().clamp_min(1.0)


def training_loss(model: SurgWMBenchDiamondDiffusion, batch: dict[str, Any], config: AdapterConfig, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    frames = batch["frames"].to(device)
    coords = batch["coords_norm"].to(device)
    mask = batch_mask(batch, device)
    bsz = coords.shape[0]
    step = torch.randint(0, config.diffusion_steps, (bsz,), device=device)
    sigma = (step.to(torch.float32) + 1.0) / float(config.diffusion_steps)
    noise = torch.randn_like(coords)
    noisy = (coords + sigma.view(-1, 1, 1) * noise).clamp(0.0, 1.0)
    output = model(frames, batch["frame_indices"].to(device), batch["num_frames"].to(device), noisy, step, mask)
    clean_loss = weighted_mse(output["clean_coords"], coords, mask, batch)
    noise_loss = weighted_mse(output["pred_noise"], noise, mask, batch)
    recon_mask = mask[:, :, None, None, None].to(dtype=torch.float32)
    pixels_per_frame = frames.shape[2] * frames.shape[3] * frames.shape[4]
    recon_loss = ((output["recon"] - frames).square() * recon_mask).sum() / (
        recon_mask.sum().clamp_min(1.0) * pixels_per_frame
    )
    loss = config.clean_weight * clean_loss + config.noise_weight * noise_loss + config.recon_weight * recon_loss
    return loss, {"loss": float(loss.item()), "clean_loss": float(clean_loss.item()), "noise_loss": float(noise_loss.item()), "recon_loss": float(recon_loss.item())}


def _metric_dict(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    return {
        "ade": traj_metrics.ade(pred, target, mask),
        "fde": traj_metrics.fde(pred, target, mask),
        "frechet": traj_metrics.discrete_frechet(pred, target, mask),
        "hausdorff": traj_metrics.symmetric_hausdorff(pred, target, mask),
        "endpoint_error": traj_metrics.endpoint_error(pred, target, mask),
        "trajectory_length_error": traj_metrics.trajectory_length_error(pred, target, mask),
        "smoothness": traj_metrics.trajectory_smoothness(pred, mask),
    }


@torch.no_grad()
def evaluate_model(model: SurgWMBenchDiamondDiffusion, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    difficulties: list[str] = []
    for batch in loader:
        mask = batch_mask(batch, device)
        output = model(batch["frames"].to(device), batch["frame_indices"].to(device), batch["num_frames"].to(device), mask=mask)
        preds.append(output["clean_coords"].cpu())
        targets.append(batch["coords_norm"].cpu())
        masks.append(mask.cpu())
        difficulties.extend(["null" if item is None else str(item) for item in batch.get("difficulty", [])])
    if not preds:
        return {"metrics_overall": {}, "metrics_by_difficulty": {}, "num_clips": 0}
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    mask = torch.cat(masks, dim=0)
    by_difficulty: dict[str, dict[str, float]] = {}
    for difficulty in ("low", "medium", "high", "null"):
        indices = [idx for idx, value in enumerate(difficulties) if value == difficulty]
        if indices:
            by_difficulty[difficulty] = _metric_dict(pred[indices], target[indices], mask[indices])
    return {"metrics_overall": _metric_dict(pred, target, mask), "metrics_by_difficulty": by_difficulty, "num_clips": int(pred.shape[0])}


def train_adapter(args: Any) -> dict[str, Any]:
    config = AdapterConfig(
        target=args.target,
        interpolation_method=args.interpolation_method,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        diffusion_steps=args.diffusion_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        clean_weight=args.clean_weight,
        noise_weight=args.noise_weight,
        recon_weight=args.recon_weight,
        num_workers=args.num_workers,
        max_clips=args.max_clips,
        max_frames=args.max_frames,
        seed=args.seed,
    )
    set_seed(config.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(args.dataset_root, args.train_manifest or args.manifest, config, shuffle=True)
    val_loader = make_loader(args.dataset_root, args.val_manifest, config, shuffle=False) if args.val_manifest else None
    model = SurgWMBenchDiamondDiffusion(config.latent_dim, config.hidden_dim, config.num_layers, config.num_heads, config.diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    epoch_metrics: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        steps = 0
        for batch in train_loader:
            loss, parts = training_loss(model, batch, config, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
        epoch_metrics.append({"epoch": epoch, **{key: value / max(steps, 1) for key, value in totals.items()}})
    checkpoint = output_dir / "checkpoint_last.pt"
    torch.save({"baseline": BASELINE_KEY, "model": MODEL_NAME, "config": asdict(config), "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "timestamp": timestamp()}, checkpoint)
    validation = evaluate_model(model, val_loader, device) if val_loader is not None else None
    metadata = {"dataset_name": "SurgWMBench", "baseline": BASELINE_KEY, "model": MODEL_NAME, "target": config.target, "train_manifest": args.train_manifest or args.manifest, "val_manifest": args.val_manifest, "checkpoint": str(checkpoint), "config": asdict(config), "timestamp": timestamp()}
    (output_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    train_metrics = {"epochs": epoch_metrics, "validation": validation}
    (output_dir / "train_metrics.json").write_text(json.dumps(train_metrics, indent=2), encoding="utf-8")
    return {"checkpoint": str(checkpoint), "metadata": metadata, "train_metrics": train_metrics}


def load_model_from_checkpoint(checkpoint: str | Path, device: torch.device) -> tuple[SurgWMBenchDiamondDiffusion, AdapterConfig, dict[str, Any]]:
    data = torch.load(checkpoint, map_location=device)
    config = AdapterConfig(**data["config"])
    model = SurgWMBenchDiamondDiffusion(config.latent_dim, config.hidden_dim, config.num_layers, config.num_heads, config.diffusion_steps).to(device)
    model.load_state_dict(data["model_state_dict"])
    return model, config, data


def eval_adapter(args: Any) -> dict[str, Any]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config, checkpoint_data = load_model_from_checkpoint(args.checkpoint, device)
    config.target = args.target
    config.interpolation_method = args.interpolation_method
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.max_clips = args.max_clips
    config.max_frames = args.max_frames
    loader = make_loader(args.dataset_root, args.manifest, config, shuffle=False)
    summary = evaluate_model(model, loader, device)
    meta = target_metadata(args.target)
    result = {
        "dataset_name": "SurgWMBench",
        "model": checkpoint_data.get("model", MODEL_NAME),
        "baseline": BASELINE_KEY,
        "manifest": args.manifest,
        "experiment_target": "sparse_20_anchor" if args.target == "sparse_20_anchor" else "dense_pseudo_label",
        "primary_target": meta["primary_target"],
        "dense_target": meta["dense_target"],
        "interpolation_method": args.interpolation_method,
        "rollout_mode": "offline_world_model_rollout",
        "checkpoint": str(args.checkpoint),
        "metrics_overall": summary["metrics_overall"],
        "metrics_by_difficulty": summary["metrics_by_difficulty"],
        "num_clips": summary["num_clips"],
        "timestamp": timestamp(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
