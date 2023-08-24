from typing import Optional, Union

import numpy as np

from env import GridWorld
from policy import Policy, EpsilonGreedyPolicy, GreedyPolicy
from utils.helper_functions import freeze_params, FromOthers


class Sarsa:
    def __init__(
        self,
        model: GridWorld,
        behavior_policy: Policy = EpsilonGreedyPolicy,
        target_policy: Optional[Policy] = None,
        *,
        alpha: float = 0.5,
        rng_state: Optional[Union[np.random.Generator, int]] = None,
    ):
        """
        Solves the supplied environment using SARSA.

        Parameters
        ----------
        model : python object
            Holds information about the environment to solve
            such as the reward structure and the transition dynamics.

        alpha : float
            Algorithm learning rate. Defaults to 0.5.

        target_policy : Policy
            The policy to evaluate the Q-values with. This policy would be the final policy of the converged Q-values
            and considered the 'learned policy'.
            Defaults to EpsilonGreedyPolicy.

        behavior_policy : Policy
            The policy to follow when sampling of actions based on the Q-values.
            This policy is the policy that is used to generate the trajectories and that determines which states are
            visited during execution. That's why it is considered the 'behavior policy'. Sarsa is an ON-POLICY
            algorithm and thus requires that target == behaviour.
            Defaults to EpsilonGreedyPolicy.

        rng_state : numpy.random.Generator or int
            Random number generator or seed for the random number generator.
        """
        self.model = model
        self.alpha = alpha
        self.behavior_policy = behavior_policy
        self.target_policy = (
            target_policy if target_policy is not None else behavior_policy
        )
        self.rng_state = np.random.default_rng(rng_state)
        # initialize the state-action value function and the state counts
        self.q_values = np.zeros(
            (self.model.num_states, self.model.num_actions), dtype=float
        )
        self.state_counts = np.zeros((self.model.num_states, 1))

    def run(
        self,
        max_horizon: int = 100,
        max_eps: int = 1000,
    ):
        """
        Runs the SARSA algorithm for the specified number of episodes.

        Parameters
        ----------
        max_horizon : int
            The maximum number of iterations to perform per episode.
            Defaults to 100.

        max_eps : int
            The number of episodes to run SARSA for.
            Defaults to 1000.

        Returns
        -------
        q : numpy array of shape (N, 1)
            The state-action value for the environment where N is the
            total number of states

        pi : numpy array of shape (N, 1)
            Optimal policy for the environment where N is the total
            number of states.
        """

        for i in range(max_eps):
            if i % 1000 == 0:
                print("Running episode %i." % i)

            # for each new episode, start at the given start state
            next_state = int(self.model.start_state_seq)
            # sample first e-greedy action
            next_action = self._sample_action(next_state)
            for t in range(max_horizon):
                state = next_state
                action = next_action
                # count the state visits
                self.state_counts[state] += 1

                # End episode if state is a terminal state
                if np.any(state == self.model.goal_states_seq):
                    break

                next_state = self._sample_next_state(state, action)
                next_action = self._sample_action(next_state)
                # Calculate the temporal difference and update q_values function
                self.update(t, state, action, next_state, next_action)

    def update(self, timestep: int, state, action, next_state, next_action):
        # remember that the reward R_{t+1} taking action A_t in state S_t and ending up in state S_{t+1} is stored in
        # model.reward[S_{t+1}], i.e. the reward at next_state
        self.q_values[state, action] += self.alpha * (
            self.model.reward[next_state]
            + self.model.gamma * self.q_values[next_state, next_action]
            - self.q_values[state, action]
        )

    def _sample_action(self, state: int):
        return int(self.behavior_policy.sample(self.q_values[state].reshape(1, -1)))

    def _sample_next_state(self, state, action):
        next_state = -1
        p = 0.0
        r = self.rng_state.random()
        # sample the next state according to the probability of the transition.
        # Once the cumulative probability is greater than `r` (the [0, 1]-uniformly sampled threshold),
        # the next state is selected as the threshold crossing state.
        # logically equivalent to:
        # self.rng_state.choice(self.model.num_states, p=self.model.probability[state, :, action])
        for next_state, transition_prob in enumerate(
            self.model.probability[state, :, action]
        ):
            p += transition_prob
            if r <= p:
                break
        assert next_state > -1
        return next_state


class ExpectedSarsa(Sarsa):
    def update(self, timestep: int, state, action, next_state, next_action):
        state_q_values = self.q_values[state]
        next_state_q_values = self.q_values[next_state]
        state_q_values[action] += self.alpha * (
            self.model.reward[next_state]
            + self.model.gamma
            * np.dot(
                next_state_q_values,
                self.target_policy(next_state_q_values.reshape(1, -1)).flatten(),
            )
            - state_q_values[action]
        )


class DoubleExpectedSarsa(Sarsa):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q2_values = np.zeroes_like(self.q_values, dtype=float)

    def _sample_action(self, state: int):
        return self.behavior_policy.sample(
            (self.q2_values[state] + self.q_values[state]).reshape(1, -1)
        )

    def update(self, timestep: int, state, action, next_state, next_action):
        # choose to use q_values or q2_values for the action selector and
        # q_values for the Q-evaluation of next_state or vice versa.
        action_q_values, updating_q_values = (
            (self.q_values, self.q2_values)
            if self.rng_state.random() < 0.5
            else (self.q2_values, self.q_values)
        )
        updating_q_values[state, action] += self.alpha * (
            self.model.reward[next_state]
            + self.model.gamma
            * np.dot(
                updating_q_values[next_state].reshape(1, -1),
                self.target_policy(action_q_values[next_state].reshape(1, -1)),
            )
            - updating_q_values[state, action]
        )


freeze_target_policy_arg = freeze_params(
    (
        (
            "target_policy",
            FromOthers(lambda *args, **kwargs: GreedyPolicy(kwargs["rng_state"])),
            2,
        ),
    )
)


class QLearning(ExpectedSarsa):
    """
    Q-Learning is the expected sarsa algorithm with a greedy target policy.
    """

    @freeze_target_policy_arg
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DoubleQLearning(DoubleExpectedSarsa):
    """
    Double Q-Learning is the double expected sarsa algorithm with a greedy target policy.
    """

    @freeze_target_policy_arg
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
