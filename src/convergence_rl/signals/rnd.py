import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RNDModel(nn.Module):
    def __init__(self, obs_dim: int, feature_dim: int = 32):
        super().__init__()

        self.target = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
        )

        for param in self.target.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-3)

    def update(self, obs: np.ndarray) -> float:
        obs = np.asarray(obs, dtype=np.float32)

        if obs.ndim == 1:
            obs = obs[None, :]

        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            target_features = self.target(obs_tensor)

        predictor_features = self.predictor(obs_tensor)
        loss = ((predictor_features - target_features) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())