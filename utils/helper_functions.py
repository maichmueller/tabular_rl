import functools
from math import floor
from typing import Tuple, Optional, Any

import numpy as np


def row_col_to_seq(row_col, num_cols):
    return row_col[:, 0] * num_cols + row_col[:, 1]


def seq_to_col_row(seq, num_cols):
    r = floor(seq / num_cols)
    c = seq - r * num_cols
    return np.array([[r, c]])


def create_policy_direction_arrays(model, policy):
    """
     define the policy directions
     0 - up    [0, 1]
     1 - down  [0, -1]
     2 - left  [-1, 0]
     3 - right [1, 0]
    :param policy:
    :return:
    """

    # intitialize direction arrays
    U = np.zeros((model.num_rows, model.num_cols))
    V = np.zeros((model.num_rows, model.num_cols))
    Action = model.Action
    for state in range(model.num_states - 1):
        # get index of the state
        i = tuple(model.index_to_state(state))
        # define the arrow direction
        pol = Action(int(np.argmax(policy[state])))
        if pol == Action.UP:
            U[i] = 0
            V[i] = 0.5
        elif pol == Action.DOWN:
            U[i] = 0
            V[i] = -0.5
        elif pol == Action.LEFT:
            U[i] = -0.5
            V[i] = 0
        elif pol == Action.RIGHT:
            U[i] = 0.5
            V[i] = 0

    return U, V


class FromOthers:
    def __init__(self, func):
        self.call = func

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


def freeze_params(*params: Tuple[str, Any, Optional[int]]):
    def _freeze_params(init_func):
        @functools.wraps(init_func)
        def init_wrapper(self, *args, **kwargs):
            original_args = args
            for param, default, pos_idx in params:
                if isinstance(default, FromOthers):
                    default = default(*original_args, **kwargs)

                if param in kwargs:
                    kwargs[param] = kwargs[default]
                # elif is important here, because if the user passes the param in as a kwarg then checking the arg list
                # for length will trigger the branch even at times when the param is not given as mere positional arg
                elif pos_idx is not None and len(args) > pos_idx:
                    args = (
                        args[:pos_idx],
                        default,
                        args[pos_idx + 1 :],
                    )
                else:
                    kwargs[param] = default
            return init_func(self, *args, **kwargs)

        return init_wrapper

    return _freeze_params
