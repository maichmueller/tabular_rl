from collections import namedtuple, defaultdict
from typing import Optional

import numpy as np

from env import GridWorld
from policy import StatePolicy

episode_entry = namedtuple("episode_entry", "state action reward truncated_return")


def generate_trajectory(
        model: GridWorld,
        state_policy: StatePolicy,
        discount: float,
        rng: np.random.Generator,
):
    episode, return_ = [], 0.0

    next_state = int(model.start_state_seq)
    next_action = state_policy[next_state].sample(rng)
    while not model.is_terminal(next_state):
        # S_t, A_t
        state, action = next_state, next_action
        # S_t+1
        next_state = model.transition(state, action, rng)
        # A_t+1
        next_action = state_policy[next_state].sample(rng)
        # R_t+1
        reward = model.reward[next_state]
        # G_t = \sum_{k=t}^{T} gamma^{k-t} R^{k+1}
        return_ = return_ * discount + reward
        episode.append(episode_entry(state, action, reward, return_))
    return episode, return_


def first_visit_mc_prediction(
        model: GridWorld,
        policy: StatePolicy,
        *,
        rng: np.random.Generator,
        convergence_criteria: Optional[float] = None,
        n_iters: Optional[int] = None,
        discount: float = 0.99,
):
    """
    Performs first visit Monte Carlo prediction for the given policy.

    Parameters
    ----------
    model: GridWorld,
        The model of the underlying environment.
    policy: StatePolicy,
        The policy to evaluate.
    rng: np.random.Generator,
        The random number generator to use.
    convergence_criteria: float, optional
        The convergence criteria for the algorithm. If the maximum difference between the old and new state values
        is less than this value, the algorithm will terminate. If None, the algorithm will run for n_iters iterations.
    n_iters: int, optional
        The number of iterations to run the algorithm for. If None, the algorithm will run until convergence.
    discount: float, optional
        The discount factor to use.

    Returns
    -------
    state_value: dict[int, float]
        The state values for each state in the environment under the given policy.
    """
    assert (
        convergence_criteria is not None or n_iters is not None,
        "Need to specify either convergence criteria or number of iterations.",
    )
    state_value = dict()
    returns = dict()
    for state in policy:
        state_value[state] = 0.0
        returns[state] = []

    def loop_gen():
        if n_iters is not None:
            for _ in range(n_iters):
                yield
            return n_iters
        else:
            diff = float("inf")
            iteration = 0
            while diff > convergence_criteria:
                diff = yield iteration
                iteration += 1
            return iteration

    for _ in (loop := loop_gen()):
        episode, return_ = generate_trajectory(model, policy, discount, rng)
        max_diff = -float("inf")
        seen = set()
        # the algorithm in the book iterates in reverse (i.e. t = T-1, T-2, ..., 0)
        # we are iterating forwards (t = 1, 2, ..., T-1)
        for t, entry in enumerate(episode, start=1):
            # entry = (s_t, a_t, r_t+1, \sum_{k=0}^t gamma^k r_{k+1})
            s_t = entry.state
            if s_t not in seen:
                # due to iterating forwards we have to compute what the return from S_t onwards is.
                # This return is computed via G_t = (G_0 - \sum_{k=0}^t gamma^k R_{k+1}) / gamma^t
                returns[s_t].append(
                    (return_ - entry.truncated_return) / (discount ** t)
                )
                new_state_value = np.mean(returns[s_t])
                max_diff = max(max_diff, abs(new_state_value - state_value[s_t]))
                state_value[s_t] = new_state_value
                seen.add(s_t)
        loop.send(max_diff)
    return state_value
