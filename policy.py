from abc import ABC
from typing import Dict, Iterable

import numpy as np


class ActionPolicy:
    def __init__(self, probabilities: Iterable[float]):
        if isinstance(probabilities, np.ndarray):
            assert probabilities.ndim == 1
            self.probabilities = probabilities.astype(float)
        else:
            self.probabilities = np.array(probabilities, dtype=float)

    def __len__(self):
        return len(self.probabilities)

    def __getitem__(self, index: int):
        return self.probabilities[index]

    def __iter__(self):
        return iter(self.probabilities)

    def value(self, q_values: np.ndarray):
        assert q_values.ndim == 1
        return np.dot(self.probabilities, q_values)

    def sample(self, rng: np.random.Generator):
        return rng.choice(len(self), p=self.probabilities)


class StatePolicy:
    def __init__(self, action_policies: Dict[int, ActionPolicy]):
        self.policies = action_policies

    def __len__(self):
        return len(self.policies)

    def __getitem__(self, state: int):
        return self.policies[state]

    def __iter__(self):
        return iter(self.policies)

    def items(self):
        return self.policies.items()

    def keys(self):
        return self.policies.keys()

    def value(self, q_values: np.ndarray):
        values = {}
        for state, policy in self.policies.items():
            values[state] = policy.value(q_values[state])
        return values

    def sample(self, rng: np.random.Generator):
        samples = {}
        for state, action_policy in self.policies.items():
            samples[state] = action_policy.sample(rng)
        return samples


class PolicyGenerator(ABC):
    def __call__(self, q_values: np.ndarray) -> ActionPolicy:
        raise NotImplementedError("'__call__' not implemented.")


class GreedyPolicyGenerator(PolicyGenerator):
    def __call__(self, q_values: np.ndarray) -> ActionPolicy:
        policy = np.zeros_like(q_values, dtype=float)
        policy[np.argmax(q_values)] = 1.0
        return ActionPolicy(policy)


class EpsilonGreedyPolicyGenerator(GreedyPolicyGenerator):
    def __init__(self, epsilon: float = 0.5):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, q_values: np.ndarray):
        assert q_values.ndim == 1
        greedy_policy = super().__call__(q_values).probabilities
        return ActionPolicy(
            (1 - self.epsilon) * greedy_policy + self.epsilon / len(q_values)
        )
