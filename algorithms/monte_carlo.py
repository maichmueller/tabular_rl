from collections import namedtuple, defaultdict
from enum import Enum
from typing import Optional, Callable

import numpy as np

from env import GridWorld
from policy import StatePolicy, ActionPolicy, GreedyPolicyGenerator

episode_entry = namedtuple("episode_entry", "state action reward truncated_return")


def generate_trajectory(
    model: GridWorld,
    state_policy: Callable[[int], ActionPolicy],
    discount: float,
    rng: np.random.Generator,
    random_start: bool = False,
):
    episode, return_ = [], 0.0

    next_state = int(rng.choice(model.start_states_seq))
    next_state_policy = state_policy(next_state)
    if random_start:
        next_action = rng.choice(model.num_actions)
    else:
        next_action = next_state_policy.sample(rng)
    while not model.is_terminal(next_state):
        # S_t, A_t
        state, action = next_state, next_action
        # S_t+1
        next_state = model.transition(state, action, rng)
        # A_t+1
        next_action = next_state_policy.sample(rng)
        # R_t+1
        reward = model.reward(next_state)
        # G_t = \sum_{k=t}^{T} gamma^{k-t} R^{k+1}
        return_ = return_ * discount + reward
        episode.append(episode_entry(state, action, reward, return_))
    return episode, return_


def _loop_gen(
    max_iters: Optional[int] = None, convergence_criteria: Optional[float] = None
):
    """
    Generate the loop depending on the convergence criteria or the nr of iterations (preferred if specified).

    This is done to avoid having to check the convergence criteria at every iteration if it isn't even desired.
    """
    if max_iters is not None:
        for _ in range(max_iters):
            yield
        return max_iters
    else:
        diff = float("inf")
        iteration = 0
        while diff > convergence_criteria:
            diff = yield iteration
            iteration += 1
        return iteration


class MCPredictionMode(Enum):
    EVERY_VISIT = 0
    FIRST_VISIT = 1


class Predictor:
    """
    Performs Monte Carlo prediction for the given policy.

    This algorithm evaluates the state-value function for the given policy by generating episodes from the environment
    and computing the average return for each state within the episode.
    """

    def __init__(
        self,
        model: GridWorld,
        policy: StatePolicy,
        mode: MCPredictionMode = MCPredictionMode.FIRST_VISIT,
        discount: float = 0.99,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Parameters
        ----------
        model: GridWorld,
            The model of the underlying environment.
        policy: StatePolicy,
            The policy to evaluate.
        mode : MCPredictionMode,
            The update mode to use for the algorithm. Either update states on EVERY_VISIT or only on the FIRST_VISIT in an
             episode.
        discount: float, optional
            The discount factor to use.
        rng: np.random.Generator,
            The random number generator to use.

        """
        self.model = model
        self.policy = policy
        self.mode = mode
        self.discount = discount
        self.state_value = np.zeros((model.num_states,), dtype=float)
        self.counts = self.state_value.astype(np.int16)
        self.rng = rng if rng is not None else np.random.default_rng()

    def run(
        self,
        convergence_criteria: Optional[float] = None,
        max_iters: Optional[int] = None,
    ):
        """
        Runs the Monte Carlo prediction algorithm for the given policy.

        Parameters
        ----------
        convergence_criteria: float, optional
            The convergence criteria for the algorithm. If the maximum difference between the old and new state values
            is less than this value, the algorithm will terminate. If None, it will run for max_iters iterations.
        max_iters: int, optional
            The number of iterations to run the algorithm for. If None, the algorithm will run until convergence.

        Returns
        -------
        state_value: dict[int, float]
            The state values for each state in the environment under the given policy.
        """
        assert (
            convergence_criteria is not None or max_iters is not None,
            "Need to specify either convergence criteria or number of iterations.",
        )
        for state in self.policy:
            # each state starts with an average value of 0.0 and a count of 0
            self.state_value[state] = (0.0, 0)

        for _ in (loop := _loop_gen()):
            episode, return_ = generate_trajectory(
                self.model, lambda s: self.policy[s], self.discount, self.rng
            )
            max_diff = -float("inf")
            seen = set()
            # the algorithm in the book iterates in reverse (i.e. t = T-1, T-2, ..., 0)
            # we are iterating forwards (t = 1, 2, ..., T-1)
            for t, entry in enumerate(episode, start=1):
                # entry = (s_t, a_t, r_t+1, \sum_{k=0}^t gamma^k r_{k+1})
                s_t = entry.state
                if self.mode == MCPredictionMode.EVERY_VISIT or s_t not in seen:
                    # due to iterating forwards we have to compute what the return from S_t onwards is.
                    # This return is computed via G_t = (G_0 - \sum_{k=0}^t gamma^k R_{k+1}) / gamma^t
                    value_incr = (return_ - entry.truncated_return) / (
                        self.discount**t
                    )
                    value, count = self.state_value[s_t]
                    new_value = (value * count + value_incr) / (count + 1)
                    self.state_value[s_t] = new_value, count + 1

                    max_diff = max(max_diff, abs(new_value - value))
                    seen.add(s_t)
            loop.send(max_diff)
        return self.state_value


class ExploringStarts:
    def __init__(
        self,
        model: GridWorld,
        mode: MCPredictionMode = MCPredictionMode.FIRST_VISIT,
        discount: float = 0.99,
        rng: Optional[np.random.Generator] = None,
    ):
        self.model = model
        self.behavior_policy = GreedyPolicyGenerator()
        self.q_values = np.zeros((model.num_states, model.num_actions), dtype=float)
        self.counts = self.q_values.astype(np.int16)
        self.mode = mode
        self.discount = discount
        self.rng = rng

    def run(
        self,
        convergence_criteria: Optional[float] = None,
        max_iters: Optional[int] = None,
    ):
        """
        Runs the Monte Carlo prediction algorithm for the given policy.

        Parameters
        ----------
        convergence_criteria: float, optional
            The convergence criteria for the algorithm. If the maximum difference between the old and new state values
            is less than this value, the algorithm will terminate. If None, it will run for max_iters iterations.
        max_iters: int, optional
            The number of iterations to run the algorithm for. If None, the algorithm will run until convergence.

        Returns
        -------
        state_value: dict[int, float]
            The state values for each state in the environment under the given policy.
        """
        assert (
            convergence_criteria is not None or max_iters is not None,
            "Need to specify either convergence criteria or number of iterations.",
        )

        for _ in (loop := _loop_gen()):
            episode, return_ = generate_trajectory(
                self.model,
                lambda s: self.behavior_policy(self.q_values[s]),
                self.discount,
                self.rng,
            )
            max_diff = -float("inf")
            seen = set()
            # the algorithm in the book iterates in reverse (i.e. t = T-1, T-2, ..., 0)
            # we are iterating forwards (t = 1, 2, ..., T-1)
            for t, entry in enumerate(episode, start=1):
                # entry = (s_t, a_t, r_t+1, \sum_{k=0}^t gamma^k r_{k+1})
                s_t, a_t = entry.state, entry.action
                if self.mode == MCPredictionMode.EVERY_VISIT or (s_t, a_t) not in seen:
                    # due to iterating forwards we have to compute what the return from S_t onwards is.
                    # This return is computed via G_t = (G_0 - \sum_{k=0}^t gamma^k R_{k+1}) / gamma^t
                    value_incr = (return_ - entry.truncated_return) / (
                        self.discount**t
                    )
                    value, count = self.q_values[s_t, a_t]
                    new_value = (value * count + value_incr) / (count + 1)
                    self.q_values[s_t, a_t] = new_value, count + 1
                    max_diff = max(max_diff, abs(new_value - value))
                    seen.add((s_t, a_t))
            loop.send(max_diff)
        return self.q_values


class OnPolicyControl:
    def __init__(
        self,
        model: GridWorld,
        mode: MCPredictionMode = MCPredictionMode.FIRST_VISIT,
        discount: float = 0.99,
        rng: Optional[np.random.Generator] = None,
    ):
        self.model = model
        self.behavior_policy = GreedyPolicyGenerator()
        self.q_values = np.zeros((model.num_states, model.num_actions), dtype=float)
        self.counts = self.q_values.astype(np.int16)
        self.mode = mode
        self.discount = discount
        self.rng = rng
