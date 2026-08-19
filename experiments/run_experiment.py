import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from convergence_rl.convergence.detector import ConvergenceLoggingCallback
from convergence_rl.environments.make_env import make_env


def run_experiment(seed: int, use_auto_stop: bool) -> None:
    env_id = "CartPole-v1"
    max_timesteps = 30_000

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(env_id=env_id, seed=seed)
    eval_env = make_env(env_id=env_id, seed=seed + 100)

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