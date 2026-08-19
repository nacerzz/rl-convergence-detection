import numpy as np
import torch


def compute_policy_entropy(model, obs: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=np.float32)

    if obs.ndim == 1:
        obs = obs[None, :]

    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(model.device)

    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        entropy = distribution.distribution.entropy().mean()

    return float(entropy.item())