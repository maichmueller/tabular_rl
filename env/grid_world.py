import itertools
from enum import Enum
from functools import singledispatchmethod
from typing import Optional, Union

import numpy as np

from utils.helper_functions import row_col_to_seq
from utils.helper_functions import seq_to_col_row


class GridWorld:
    """
    Creates a gridworld object to pass to an RL algorithm.

    Parameters
    ----------
    num_rows : int
        The number of rows in the gridworld.

    num_cols : int
        The number of cols in the gridworld.

    start_states : numpy array of shape (1, 2), np.array([[row, col]])
        The start state of the gridworld (can only be one start state)

    goal_states : numpy array of shape (n, 2)
        The goal states for the gridworld where n is the number of goal
        states.
    """

    class Action(Enum):
        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3

    num_actions: int
    num_states: int
    num_restart_states: int
    _reward: np.ndarray
    _probability: np.ndarray

    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        start_states: np.ndarray,
        goal_states: np.ndarray,
        goal_reward: Union[float, np.ndarray],
        step_reward: Union[float, np.ndarray],
        p_transition_success: float = 1.0,
        bias: float = 0.0,
        obstacle_states: Optional[np.ndarray] = None,
        bad_states: Optional[np.ndarray] = None,
        bad_state_reward: Optional[Union[float, np.ndarray]] = None,
        restart_states: Optional[np.ndarray] = None,
        restart_state_reward: Optional[Union[float, np.ndarray]] = None,
    ):
        """

        Parameters
        ----------
        num_rows: int
            The number of rows in the gridworld.
        num_cols: int
            The number of cols in the gridworld.
        start_states: np.ndarray, shape (k, 2)
            The start states of the gridworld.
        goal_states: np.ndarray, shape (m, 2)
            The goal states for the gridworld where m <= n is the number of goal states.
        step_reward: float
            The reward for each step taken by the agent in the grid world.
            Typically, a negative value (e.g. -1).
        goal_reward: float
            The reward given to the agent for reaching the goal state.
            Typically, a middle range positive value (e.g. 10)
        bad_state_reward: float
            The reward given to the agent for transitioning to a bad state.
            Typically, a middle range negative value (e.g. -6)
        restart_state_reward: float
            The reward given to the agent for transitioning to a restart state.
            Typically, a large negative value (e.g. -100)
        p_transition_success : float (in the interval [0,1])
            The probability that the agents attempted transition is successful.
            p_transition_success is the probability that the agent successfully
            executes the intended action. The action is then incorrectly executed
            with probability 1 - p_good_transition and in this case the agent
            transitions to the left of the intended transition with probability
            (1 - p_transition_success) * bias and to the right with probability
            (1 - p_transition_success) * (1 - bias).
        bias : float (in the interval [0,1])
            The probability that the agent transitions left or right of the intended
            transition if the intended transition is not successful.
        obstacle_states : np.ndarray, shape (n, 2)
            States the agent cannot enter where n is the number of obstacle states
            and the two columns are the row and col position of the obstacle state.
        bad_states: np.ndarray, shape (n, 2)
            States in which the agent incurs high penalty where n is the number of bad
            states and the two columns are the row and col position of the bad state.
        restart_states: np.ndarray, shape (n, 2)
            States in which the agent incurs high penalty and transitions to the start
            state where n is the number of restart states and the two columns are the
            row and col position of the restart state.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_actions = 4
        self.num_states = self.num_cols * self.num_rows
        self.start_states = start_states
        self.goal_states = goal_states
        self.start_states_seq: np.ndarray = row_col_to_seq(
            self.start_states, self.num_cols
        )  # shape (k, 2)
        self.goal_states_seq: np.ndarray = row_col_to_seq(
            self.goal_states, self.num_cols
        )  # shape (k, 2)
        self.obs_states = obstacle_states
        self.bad_states = bad_states

        self.num_bad_states = bad_states.shape[0] if bad_states is not None else 0

        self.restart_states = restart_states

        self.num_restart_states = (
            restart_states.shape[0] if restart_states is not None else 0
        )

        self.p_transition_success = p_transition_success
        self.bias = bias
        self.r_step = step_reward
        self.r_goal = goal_reward
        self.r_bad = bad_state_reward
        self.r_restart = restart_state_reward
        self._make()

    def _make(self):
        """
        Create the grid world with the specified parameters.
        """

        # rewards structure
        self._reward = self.r_step * np.ones((self.num_states, 1))
        self._reward[self.num_states - 1] = 0
        self._reward[self.goal_states_seq] = self.r_goal.reshape(-1, 1)
        for i in range(self.num_bad_states):
            if self.r_bad is None:
                raise Exception("Bad state specified but no reward is given")
            bad_state = row_col_to_seq(
                self.bad_states[i, :].reshape(1, -1), self.num_cols
            )
            self._reward[bad_state, :] = self.r_bad
        for i in range(self.num_restart_states):
            if self.r_restart is None:
                raise Exception("Restart state specified but no reward is given")
            restart_state = row_col_to_seq(
                self.restart_states[i, :].reshape(1, -1), self.num_cols
            )
            self._reward[restart_state, :] = self.r_restart

        # probability model
        self._probability = np.zeros(
            (self.num_states, self.num_actions, self.num_states)
        )
        for action, state in itertools.product(
            range(self.num_actions), range(self.num_states)
        ):
            # check if state is the fictional end state - self transition
            if state == self.num_states - 1:
                self._probability[state, action, state] = 1.0
                continue

            # check if the state is the goal state or an obstacle state - transition to end
            row_col = seq_to_col_row(state, self.num_cols)
            if self.obs_states is not None:
                end_states = np.vstack((self.obs_states, self.goal_states))
            else:
                end_states = self.goal_states

            if np.any(np.sum(np.abs(end_states - row_col), 1) == 0):
                self._probability[state, action, self.num_states - 1] = 1

            # else consider stochastic effects of action
            else:
                for axis in range(-1, 2, 1):
                    direction = self.get_direction(action, axis)
                    next_state = self._get_state(state, direction)
                    if axis == 0:
                        prob = self.p_transition_success
                    elif axis == -1:
                        prob = (1 - self.p_transition_success) * self.bias
                    else:  # axis == 1 then
                        prob = (1 - self.p_transition_success) * (1 - self.bias)

                    self._probability[state, action, next_state] += prob

            # make restart states transition back to the start states with probability 1
            if self.restart_states is not None:
                if np.any(np.sum(np.abs(self.restart_states - row_col), 1) == 0):
                    next_state = row_col_to_seq(self.start_states, self.num_cols)
                    self._probability[state, :, :] = 0
                    self._probability[state, :, next_state] = 1

    def __len__(self):
        return self.num_states

    @singledispatchmethod
    def is_terminal(self, state: int):
        return np.isin(state, self.goal_states_seq)

    @is_terminal.register(np.ndarray)
    def _(self, state: np.ndarray):
        return np.isin(state, self.goal_states)

    def transition(self, state, action, rng):
        next_state = -1
        p = 0.0
        r = rng.random()
        # sample the next state according to the probability of the transition.
        # Once the cumulative probability is greater than `r` (the [0, 1]-uniformly sampled threshold),
        # the next state is selected as the threshold crossing state.
        # logically equivalent to:
        # self.rng_state.choice(self.model.num_states, p=self.model.probability[state, :, action])
        for next_state, transition_prob in enumerate(
            self._probability[state, action, :]
        ):
            p += transition_prob
            if r <= p:
                break
        assert next_state > -1
        return next_state

    def state_to_index(self, state: np.ndarray):
        return state[0] * self.num_cols + state[1]

    def index_to_state(self, state: int):
        return np.array([state // self.num_cols, state % self.num_cols])

    @singledispatchmethod
    def reward(self, state: np.ndarray):
        return float(self._reward[self.state_to_index(state)])

    @reward.register(int)
    def _(self, state: int):
        return float(self._reward[state])

    @staticmethod
    def get_direction(action, direction):
        """
        Takes in a direction and an action and returns a new direction.

        Parameters
        ----------
        action : int
            The current action 0, 1, 2, 3 for gridworld.

        direction : int
            Either -1, 0, 1.

        Returns
        -------
        direction : int
            Value either 0, 1, 2, 3.
        """
        left = [2, 3, 1, 0]
        right = [3, 2, 0, 1]
        if direction == 0:
            new_direction = action
        elif direction == -1:
            new_direction = left[action]
        elif direction == 1:
            new_direction = right[action]
        else:
            raise Exception("get_direction received an unspecified case")
        return new_direction

    def _get_state(self, state, direction):
        """
        Get the next_state from the current state and a direction.

        Parameters
        ----------
        state : int
            The current state.

        direction : int
            The current direction.

        Returns
        -------
        next_state : int
            The next state given the current state and direction.
        """
        row_change = [-1, 1, 0, 0]
        col_change = [0, 0, -1, 1]
        row_col = seq_to_col_row(state, self.num_cols)
        row_col[0, 0] += row_change[direction]
        row_col[0, 1] += col_change[direction]

        # check for invalid states
        if self.obs_states is not None:
            if (
                np.any(row_col < 0)
                or np.any(row_col[:, 0] > self.num_rows - 1)
                or np.any(row_col[:, 1] > self.num_cols - 1)
                or np.any(np.sum(abs(self.obs_states - row_col), 1) == 0)
            ):
                next_state = state
            else:
                next_state = row_col_to_seq(row_col, self.num_cols)[0]
        else:
            if (
                np.any(row_col < 0)
                or np.any(row_col[:, 0] > self.num_rows - 1)
                or np.any(row_col[:, 1] > self.num_cols - 1)
            ):
                next_state = state
            else:
                next_state = row_col_to_seq(row_col, self.num_cols)[0]

        return next_state
