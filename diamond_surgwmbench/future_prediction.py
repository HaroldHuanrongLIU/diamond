from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "surgwmbench_benchmark").is_dir():
        sys.path.insert(0, str(parent))
        break
    if (parent / "src").is_dir():
        sys.path.insert(0, str(parent / "src"))

import torch
from torch import nn

from diamond_surgwmbench.adapter import SurgWMBenchDiamondDiffusion
from surgwmbench_benchmark.future_model_helpers import normalized_context_time, normalized_future_time
from surgwmbench_benchmark.future_prediction import FutureProtocolConfig, main, resolved_context_horizon


def _load_inner_model_classes():
    """Load DIAMOND's diffusion inner model without importing the heavy package init."""

    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
    diffusion_dir = root / "src" / "models" / "diffusion"
    package_name = "src.models.diffusion"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(diffusion_dir)]
        sys.modules[package_name] = package
    module_name = f"{package_name}.inner_model"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, diffusion_dir / "inner_model.py")
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load DIAMOND inner_model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.InnerModel, module.InnerModelConfig


class DIAMONDFuturePredictionModel(nn.Module):
    """Future-prediction wrapper around the DIAMOND diffusion world-model core."""

    def __init__(self, config: FutureProtocolConfig) -> None:
        super().__init__()
        self.core = SurgWMBenchDiamondDiffusion(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=2,
            num_heads=4,
            diffusion_steps=16,
        )
        context_frames, _ = resolved_context_horizon(config)
        cond_channels = max(32, context_frames * 16)
        if cond_channels % context_frames != 0:
            cond_channels += context_frames - (cond_channels % context_frames)
        inner_model_cls, inner_model_config_cls = _load_inner_model_classes()
        self.diffusion_core = inner_model_cls(
            inner_model_config_cls(
                img_channels=3,
                num_steps_conditioning=context_frames,
                cond_channels=cond_channels,
                depths=[1, 1],
                channels=[32, 64],
                attn_depths=[False, True],
                num_actions=1,
            )
        )
        self.context_frames = context_frames
        self.hidden_to_latent = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        frames = batch["context_frames"]
        bsz, context, channels, height, width = frames.shape
        z = self.core.encoder(frames.reshape(bsz * context, channels, height, width)).view(bsz, context, -1)
        future_z = z[:, -1:].expand(-1, batch["future_frame_indices"].shape[1], -1)
        model_in = torch.cat(
            [
                self.core.input_proj(torch.cat([z, normalized_context_time(batch)], dim=-1)),
                self.core.input_proj(torch.cat([future_z, normalized_future_time(batch)], dim=-1)),
            ],
            dim=1,
        )
        hidden = self.core.transformer(model_in)[:, context:]
        pred_coords = torch.sigmoid(self.core.clean_head(hidden))
        obs_context = frames[:, -self.context_frames :].reshape(bsz, self.context_frames * channels, height, width)
        noisy = frames[:, -1]
        act = torch.zeros(bsz, self.context_frames, dtype=torch.long, device=frames.device)
        c_noise = torch.zeros(bsz, dtype=frames.dtype, device=frames.device)
        future_frames = []
        for _ in range(batch["future_frame_indices"].shape[1]):
            residual = self.diffusion_core(noisy, c_noise, obs_context, act)
            noisy = (noisy - residual).clamp(0.0, 1.0)
            future_frames.append(noisy)
            obs_context = torch.cat([obs_context[:, channels:], noisy], dim=1)
        pred_frames = torch.stack(future_frames, dim=1)
        return {"pred_frames": pred_frames, "pred_coords_norm": pred_coords}


def make_model(config: FutureProtocolConfig) -> nn.Module:
    return DIAMONDFuturePredictionModel(config)


if __name__ == "__main__":
    raise SystemExit(main("diamond", "DIAMONDFuturePredictionCore", "diamond_surgwmbench.data.surgwmbench", make_model))
