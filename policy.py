from abc import ABC
from typing import Optional

import numpy as np


class Policy(ABC):
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self, q_values: np.ndarray):
        raise NotImplementedError("'__call__' not implemented.")

    def state_value(self, q_values: np.ndarray):
        return np.dot(self(q_values), q_values, axis=1).reshape(-1, 1)

    def sample(self, q_values: np.ndarray):
        # rng.choice() does not support 2D arrays, so we need to create the choice mechanism ourselves
        distributions = self(q_values)  # (n_states, n_actions)
        cumul_weights = np.cumsum(distributions, axis=1)  # (n_states, n_actions)
        # sample from the cumulative weights
        return (self.rng.random(q_values.shape[0], 1) < cumul_weights).argmax(axis=1)


class GreedyPolicy(Policy):
    def __call__(self, q_values: np.ndarray):
        assert q_values.ndim == 2
        policy = np.zeros_like(q_values, dtype=float)
        policy[np.arange(policy.shape[0]), np.argmax(q_values, axis=1)] = 1.0
        return policy

    def state_value(self, q_values: np.ndarray):
        assert q_values.ndim == 2
        return np.max(q_values, axis=1).reshape(-1, 1)

    def sample(self, q_values: np.ndarray):
        return np.argmax(q_values, axis=1)


class EpsilonGreedyPolicy(GreedyPolicy):
    def __init__(self, *args, epsilon: float = 0.5):
        super().__init__(*args)
        self.epsilon = epsilon

    def __call__(self, q_values: np.ndarray):
        assert q_values.ndim == 2
        greedy_policy = super().__call__(q_values)
        return (1 - self.epsilon) * greedy_policy + self.epsilon / q_values.shape[1]

    def state_value(self, q_values: np.ndarray):
        assert q_values.ndim == 2
        greedy_value = super().state_value(q_values)
        # uniform expectation is simply the mean
        return (1 - self.epsilon) * greedy_value + self.epsilon * np.mean(
            q_values, axis=1
        ).reshape(-1, 1)

    def sample(self, q_values: np.ndarray):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, q_values.shape[1])
        else:
            return super().sample(q_values)
