from math import floor
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
        i = tuple(seq_to_col_row(state, model.num_cols)[0])
        # define the arrow direction
        pol = Action(int(policy[state]))
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
