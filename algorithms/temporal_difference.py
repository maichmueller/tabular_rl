from typing import Optional, Union

import numpy as np

from env import GridWorld
from policy import PolicyGenerator, EpsilonGreedyPolicyGenerator, GreedyPolicyGenerator
from utils.helper_functions import freeze_params


class Sarsa:
    def __init__(
        self,
        model: GridWorld,
        behavior_policy: PolicyGenerator = EpsilonGreedyPolicyGenerator,
        target_policy: Optional[PolicyGenerator] = None,
        *,
        alpha: float = 0.5,
        discount=0.99,
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

        target_policy : PolicyGenerator
            The policy which we aim to generate from the converged Q-values and thus consider the 'learned policy'.
            It is also the policy to evaluate the next-state's Q-values with.
            Defaults to EpsilonGreedyPolicy.

        behavior_policy : PolicyGenerator
            The policy to follow when sampling actions based on the Q-values to navigate the environment.
            This policy is the policy that is used to generate the trajectories and that determines which states are
            visited during execution, i.e. the behavior. Sarsa is an ON-POLICY algorithm and thus target == behaviour.
            If a differing choice of target and behavior policy is given, then the method becomes OFF-POLICY.
            Defaults to EpsilonGreedyPolicy.

        rng_state : numpy.random.Generator or int
            Random number generator or seed for the random number generator.
        """
        self.model = model
        self.alpha = alpha
        self.discount = discount
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
        Runs the SARSA algorithm for the specified number of episodes and maximum horizon in each episode.

        Parameters
        ----------
        max_horizon : int
            The maximum number of iterations to perform per episode.
            Defaults to 100.

        max_eps : int
            The number of episodes to run SARSA for.
            Defaults to 1000.
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
                if self.model.is_terminal(state):
                    break

                next_state = self.model.transition(state, action, self.rng_state)
                next_action = self._sample_action(next_state)
                # Calculate the temporal difference and update q_values function
                self.update(t, state, action, next_state, next_action)

    def update(self, timestep: int, state, action, next_state, next_action):
        # remember that the reward R_{t+1} taking action A_t in state S_t and ending up in state S_{t+1} is stored in
        # model.reward[S_{t+1}], i.e. the reward at next_state
        self.q_values[state, action] += self.alpha * (
            self.model.reward[next_state]
            + self.discount * self.q_values[next_state, next_action]
            - self.q_values[state, action]
        )

    def _sample_action(self, state: int):
        return int(self.behavior_policy(self.q_values[state]).sample(self.rng_state))


class ExpectedSarsa(Sarsa):
    """
    Solves the environment using the Expected SARSA algorithm.

    This algorithm differs from basic SARSA in that it does not sample the next action from the behavior policy but
    instead uses the expected value of the next state's Q-values under the target policy.
    """

    def update(self, timestep: int, state, action, next_state, next_action):
        state_q_values = self.q_values[state]
        next_state_q_values = self.q_values[next_state]
        state_q_values[action] += self.alpha * (
            self.model.reward[next_state]
            + self.discount
            * np.dot(
                next_state_q_values,
                self.target_policy(next_state_q_values).probabilities,
            )
            - state_q_values[action]
        )


class DoubleExpectedSarsa(Sarsa):
    """
    Solves the environment using the Double Expected SARSA algorithm.

    This algorithm differs from Expected SARSA in that it uses two Q-value functions to estimate the next state's
    Q-values. The action is sampled from one of the Q-value functions and the Q-values of the next state are evaluated
    using the other Q-value function. Which Q-value function is used for which purpose is chosen randomly anew for each
    update.
    Likewise, actions are sampled according to the aggregated Q-values of both Q-value functions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q2_values = np.zeros_like(self.q_values, dtype=float)

    def _sample_action(self, state: int):
        return int(
            self.behavior_policy((self.q2_values[state] + self.q_values[state])).sample(
                self.rng_state
            )
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
            + self.discount
            * np.dot(
                updating_q_values[next_state],
                self.target_policy(action_q_values[next_state]).probabilities,
            )
            - updating_q_values[state, action]
        )


freeze_target_policy_arg = freeze_params(
    (
        "target_policy",
        GreedyPolicyGenerator(),
        2,
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
