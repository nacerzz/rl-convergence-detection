from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from convergence_rl.convergence.stopping_rules import has_converged
from convergence_rl.evaluation.evaluate import evaluate_agent
from convergence_rl.logging.metrics_logger import save_records
from convergence_rl.signals.policy_entropy import compute_policy_entropy
from convergence_rl.signals.rnd import RNDModel
from convergence_rl.signals.state_visitation import StateVisitationCounter


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
        self.state_counter = StateVisitationCounter()

        self.records = []
        self.recent_rnd_errors = deque(maxlen=eval_freq)
        self.recent_policy_entropies = deque(maxlen=eval_freq)
        self.recent_visited_states = deque(maxlen=eval_freq)

    def _on_step(self) -> bool:
        new_obs = self.locals.get("new_obs", None)

        if new_obs is not None:
            rnd_error = self.rnd.update(new_obs)
            entropy = compute_policy_entropy(self.model, new_obs)
            visited_states = self.state_counter.update(new_obs)

            self.recent_rnd_errors.append(rnd_error)
            self.recent_policy_entropies.append(entropy)
            self.recent_visited_states.append(visited_states)

        if self.num_timesteps % self.eval_freq == 0:
            mean_eval_return, std_eval_return = evaluate_agent(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
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

            visited_states = (
                int(self.recent_visited_states[-1])
                if len(self.recent_visited_states) > 0
                else 0
            )

            record = {
                "seed": self.seed,
                "timesteps": self.num_timesteps,
                "eval_return_mean": mean_eval_return,
                "eval_return_std": std_eval_return,
                "rnd_prediction_error": mean_rnd_error,
                "policy_entropy": mean_policy_entropy,
                "visited_states": visited_states,
                "auto_stop": self.use_auto_stop,
                "stopped": False,
            }

            self.records.append(record)

            print(
                f"seed={self.seed} | steps={self.num_timesteps} | "
                f"return={mean_eval_return:.2f} | "
                f"rnd={mean_rnd_error:.6f} | "
                f"entropy={mean_policy_entropy:.4f} | "
                f"visited={visited_states}"
            )

            save_records(self.records, self.result_path)

            if self.use_auto_stop and has_converged(
                records=self.records,
                patience=self.patience,
                min_mean_return=self.min_mean_return,
                max_mean_rnd_error=self.max_mean_rnd_error,
                max_entropy_change=self.max_entropy_change,
            ):
                print(f"Automatic stopping triggered at step {self.num_timesteps}")
                self.records[-1]["stopped"] = True
                save_records(self.records, self.result_path)
                return False

        return True

    def _on_training_end(self) -> None:
        save_records(self.records, self.result_path)