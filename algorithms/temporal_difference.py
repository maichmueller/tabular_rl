from enum import Enum
from typing import Optional, Union, Callable, Iterable

import numpy as np

from env import GridWorld


class SampleMode(Enum):
    """
    Enum class for the sample mode.
    """

    ON_POLICY = 0
    GREEDY = 1


class Sarsa:
    def __init__(
        self,
        model: GridWorld,
        alpha: float = 0.5,
        epsilon: float = 0.1,
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

        epsilon : float
             Probability that a random action is selected. epsilon must be
             in the interval [0,1] where 0 means that the action is selected
             in a completely greedy manner and 1 means the action is always
             selected randomly.

        rng_state : numpy.random.Generator or int
            Random number generator or seed for the random number generator.
        """
        self.model = model
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng_state = np.random.default_rng(rng_state)
        # initialize the state-action value function and the state counts
        self.q_values = np.zeros((self.model.num_states, self.model.num_actions))
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
            next_action = self.sample_action(next_state)
            for t in range(max_horizon):
                state = next_state
                action = next_action
                # count the state visits
                self.state_counts[state] += 1

                # End episode if state is a terminal state
                if np.any(state == self.model.goal_states_seq):
                    break

                next_state = self._sample_next_state(state, action)
                # epsilon-greedy action selection
                next_action = self.sample_action(next_state)
                # Calculate the temporal difference and update q_values function
                self.update(t, state, action, next_state, next_action)

        # determine the value function with respect to the greedy policy and policy
        greedy_policy = np.argmax(self.q_values, axis=1).reshape(-1, 1)

        greedy_value_function = self.q_values[
            np.arange(self.q_values.shape[0]), greedy_policy.flatten()
        ].reshape(-1, 1)

        return greedy_value_function, greedy_policy

    def _sample_next_state(self, state, action):
        next_state = -1
        p = 0.0
        r = self.rng_state.random()
        # sample the next state according to the action and the
        # probability of the transition. Once the cumulative probability is greater than r (the [0, 1]-uniformly sampled
        # threshold), the next state is selected as the threshold crossing state.
        for next_state, transition_prob in enumerate(
            self.model.probability[state, :, action]
        ):
            p += transition_prob
            if r <= p:
                break
        return next_state

    def update(self, timestep: int, state, action, next_state, next_action):
        # remember that the reward R_{t+1} taking action A_t in state S_t and ending up in state S_{t+1} is stored in
        # model.reward[S_{t+1}], i.e. the reward at next_state
        self.q_values[state, action] += self.alpha * (
            self.model.reward[next_state]
            + self.model.gamma * self.q_values[next_state, next_action]
            - self.q_values[state, action]
        )

    def derive_policy(self, q_values: Optional[np.ndarray] = None):
        if q_values is None:
            q_values = self.q_values
        return np.argmax(q_values, axis=1).reshape(-1, 1)

    def sample_action(
        self,
        state: int,
        mode: SampleMode = SampleMode.GREEDY,
    ):
        """
        Epsilon greedy action selection.

        Parameters
        ----------
        state : int
            The current state.

        mode : SampleMode
            The sample mode to use. Defaults to SampleMode.GREEDY.

        Returns
        -------
        action : int
            Number representing the selected action between 0 and num_actions.
        """
        if self.rng_state.random() < self.epsilon:
            action = self.rng_state.integers(0, self.model.num_actions)
        else:
            if mode == SampleMode.ON_POLICY:
                action = self.rng_state.choice(
                    self.model.num_actions, p=self.q_values[state, :]
                )
            elif mode == SampleMode.GREEDY:
                action = np.argmax(self.q_values[state, :])
            else:
                raise ValueError("Unknown sample mode.")
        return action


class QLearning(Sarsa):
    def update(self, timestep: int, state, action, next_state, next_action):
        self.q_values[state, action] += self.alpha * (
            self.model.reward[next_state]
            + self.model.gamma * np.max(self.q_values[next_state, :])
            - self.q_values[state, action]
        )


class ExpectedSarsa(Sarsa):
    def __init__(
        self,
        *args,
        behaviour_policy: Optional[Callable[[int], Iterable[float]]] = None,
        **kwargs,
    ):
        """
        Expected SARSA algorithm.


        Parameters
        ----------
        behaviour_policy : Optional[Callable[[int], Iterable[float]]],
            A function that takes a state (as sequence id) as input and returns a probability distribution over actions.
            If None, the behaviour policy is the same as the target policy, in this case expected sarsa is ON-POLICY.
            The target policy in this case is the greedy policy with respect to the learned Q-values.
            If a behaviour policy is provided, expected sarsa is OFF-POLICY, i.e. the update rule will consider the
            expectation of the next state's Q-value under this behaviour policy.
        """

        self.behaviour_policy = behaviour_policy
        super().__init__(*args, **kwargs)

    def update(self, timestep: int, state, action, next_state, next_action):
        if self.behaviour_policy is None:
            behaviour_prob = np.argmax(self.model.num_actions)
            behaviour_prob[next_action] = 1.0
        else:
            behaviour_prob = self.behaviour_policy(next_state)
        self.q_values[state, action] += self.alpha * (
            self.model.reward[next_state]
            + self.model.gamma * np.sum(self.q_values[next_state, :] * behaviour_prob)
            - self.q_values[state, action]
        )
