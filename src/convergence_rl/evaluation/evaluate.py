from stable_baselines3.common.evaluation import evaluate_policy


def evaluate_agent(model, eval_env, n_eval_episodes: int = 5) -> tuple[float, float]:
    mean_return, std_return = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    return float(mean_return), float(std_return)