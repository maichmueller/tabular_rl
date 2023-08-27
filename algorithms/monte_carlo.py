from collections import namedtuple, defaultdict

import numpy as np

from env import GridWorld
from policy import StatePolicy

episode_entry = namedtuple("episode_entry", "state action reward current_return")


def generate_trajectory(
    model: GridWorld, state_policy: StatePolicy, discount: float, rng: np.random.Generator
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
        # G_t = E[\sum_{k=t}^{T} gamma^{k-t} R^{k+1} | S_t]
        return_ = return_ * discount + reward
        episode.append(episode_entry(state, action, reward, return_))
    return episode, return_


def first_visit_mc_prediction(
    model: GridWorld, policy: StatePolicy, n_iters: int, discount: float, rng: np.random.Generator
):
    state_value = dict()
    returns = dict()
    for state in policy:
        state_value[state] = 0.0
        returns[state] = []

    for it in range(n_iters):
        episode, return_ = generate_trajectory(model, policy, discount, rng)
        seen = set()
        # the algorithm in the book iterates in reverse (i.e. t = T-1, T-2, ..., 0)
        # we are iterating forwards (t=0, 1, ..., T-1)
        for entry in episode:
            if entry.state not in seen:
                # due to iterating forwards we have to compute what the return from S_t onwards is.
                # This return is computed via G_T -
                returns[entry.state].append(return_ - entry.current_return)
