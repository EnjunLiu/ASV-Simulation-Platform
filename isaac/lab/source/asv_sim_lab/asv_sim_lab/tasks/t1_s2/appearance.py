"""Sampling of task-independent OWL appearance features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class AppearanceSampler:
    """Sample red, blue and two white features for every environment."""

    def __init__(self, path: str | Path, device: torch.device | str, seed: int = 0) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"appearance library not found: {path}; see isaac/lab/README.md runtime assets"
            )
        with np.load(path, allow_pickle=False) as blob:
            self.samples = {
                label: torch.as_tensor(np.asarray(blob[label]), device=device, dtype=torch.float32)
                for label in ("red", "blue", "white")
            }
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))

    def sample(self, count: int) -> torch.Tensor:
        features = []
        for label in ("red", "blue", "white", "white"):
            bank = self.samples[label]
            index = torch.randint(0, len(bank), (int(count),), generator=self.generator, device=self.device)
            features.append(bank[index])
        return torch.stack(features, dim=1)
