import numpy as np


class StateVisitationCounter:
    def __init__(self, decimals: int = 2):
        self.decimals = decimals
        self.visited_states = set()

    def update(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        rounded_obs = np.round(obs, self.decimals)
        state_key = tuple(rounded_obs.flatten())

        self.visited_states.add(state_key)

        return len(self.visited_states)

    def reset(self) -> None:
        self.visited_states.clear()