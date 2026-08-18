import os
from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


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


class ConvergenceLoggingCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        seed: int,
        result_path: str,
        eval_freq: int = 1000,
        n_eval_episodes: int = 5,
        use_auto_stop: bool = False,
        patience: int = 3,
        min_mean_return: float = 380.0,
        max_mean_rnd_error: float = 0.0002,
        max_entropy_change: float = 0.04,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.eval_env = eval_env
        self.seed = seed
        self.result_path = result_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.use_auto_stop = use_auto_stop
        self.patience = patience
        self.min_mean_return = min_mean_return
        self.max_mean_rnd_error = max_mean_rnd_error
        self.max_entropy_change = max_entropy_change

        obs_dim = eval_env.observation_space.shape[0]
        self.rnd = RNDModel(obs_dim)

        self.records = []
        self.recent_rnd_errors = deque(maxlen=eval_freq)
        self.recent_policy_entropies = deque(maxlen=eval_freq)

    def _get_policy_entropy(self, obs: np.ndarray) -> float:
        obs = np.asarray(obs, dtype=np.float32)

        if obs.ndim == 1:
            obs = obs[None, :]

        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.model.device)

        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            entropy = distribution.distribution.entropy().mean()

        return float(entropy.item())

    def _on_step(self) -> bool:
        new_obs = self.locals.get("new_obs", None)

        if new_obs is not None:
            rnd_error = self.rnd.update(new_obs)
            entropy = self._get_policy_entropy(new_obs)

            self.recent_rnd_errors.append(rnd_error)
            self.recent_policy_entropies.append(entropy)

        if self.num_timesteps % self.eval_freq == 0:
            mean_eval_return, std_eval_return = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )

            mean_rnd_error = (
                float(np.mean(self.recent_rnd_errors))
                if len(self.recent_rnd_errors) > 0
                else np.nan
            )

            mean_policy_entropy = (
                float(np.mean(self.recent_policy_entropies))
                if len(self.recent_policy_entropies) > 0
                else np.nan
            )

            record = {
                "seed": self.seed,
                "timesteps": self.num_timesteps,
                "eval_return_mean": mean_eval_return,
                "eval_return_std": std_eval_return,
                "rnd_prediction_error": mean_rnd_error,
                "policy_entropy": mean_policy_entropy,
                "auto_stop": self.use_auto_stop,
                "stopped": False,
            }

            self.records.append(record)

            print(
                f"seed={self.seed} | steps={self.num_timesteps} | "
                f"return={mean_eval_return:.2f} | "
                f"rnd={mean_rnd_error:.6f} | "
                f"entropy={mean_policy_entropy:.4f}"
            )

            self._save_results()

            if self.use_auto_stop and self._has_converged():
                print(f"Automatic stopping triggered at step {self.num_timesteps}")
                self.records[-1]["stopped"] = True
                self._save_results()
                return False

        return True

    def _has_converged(self) -> bool:
        if len(self.records) < self.patience:
            return False

        recent = self.records[-self.patience :]

        returns = [r["eval_return_mean"] for r in recent]
        rnd_errors = [r["rnd_prediction_error"] for r in recent]
        entropies = [r["policy_entropy"] for r in recent]

        mean_return = float(np.mean(returns))
        mean_rnd_error = float(np.mean(rnd_errors))
        entropy_change = float(max(entropies) - min(entropies))

        return (
            mean_return >= self.min_mean_return
            and mean_rnd_error <= self.max_mean_rnd_error
            and entropy_change <= self.max_entropy_change
        )

    def _save_results(self) -> None:
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        df = pd.DataFrame(self.records)
        df.to_csv(self.result_path, index=False)

    def _on_training_end(self) -> None:
        self._save_results()


def run_experiment(seed: int, use_auto_stop: bool) -> None:
    env_id = "CartPole-v1"
    max_timesteps = 30_000

    env = Monitor(gym.make(env_id))
    eval_env = Monitor(gym.make(env_id))

    result_name = "auto_stop" if use_auto_stop else "fixed_budget"
    result_path = f"results/cartpole/{result_name}_seed_{seed}.csv"

    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        verbose=0,
    )

    callback = ConvergenceLoggingCallback(
        eval_env=eval_env,
        seed=seed,
        result_path=result_path,
        eval_freq=1000,
        n_eval_episodes=5,
        use_auto_stop=use_auto_stop,
        patience=3,
        min_mean_return=380.0,
        max_mean_rnd_error=0.0002,
        max_entropy_change=0.04,
    )

    model.learn(
        total_timesteps=max_timesteps,
        callback=callback,
    )

    env.close()
    eval_env.close()


if __name__ == "__main__":
    seeds = [0, 1, 2]

    for seed in seeds:
        print(f"\nRunning fixed-budget PPO, seed {seed}")
        run_experiment(seed=seed, use_auto_stop=False)

    for seed in seeds:
        print(f"\nRunning automatic-stopping PPO, seed {seed}")
        run_experiment(seed=seed, use_auto_stop=True)

    print("\nFinished. Results are saved in results/cartpole/")