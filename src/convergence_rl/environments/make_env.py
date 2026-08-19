import gymnasium as gym
from stable_baselines3.common.monitor import Monitor


def make_env(env_id: str = "CartPole-v1", seed: int | None = None):
    env = gym.make(env_id)
    env = Monitor(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    return env