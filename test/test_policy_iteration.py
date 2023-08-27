import sys

sys.path.append("..")
import numpy as np
from env.grid_world import GridWorld
from algorithms.dynamic_programming import policy_iteration


def test_policy_iteration():
    # load the test data
    vp = np.load("../data/test_data/test_policy_iteration.npy")
    test_value = vp[:, 0].reshape(-1, 1)
    test_pi = vp[:, 1].reshape(-1, 1)

    # specify world parameters
    num_cols = 12
    num_rows = 9
    obstacles = np.array(
        [[8, 6], [7, 6], [6, 6], [5, 6], [4, 6], [3, 6], [3, 7], [3, 8], [3, 9]]
    )
    bad_states = np.array([[2, 1]])
    start_state = np.array([[0, 1]])
    goal_state = np.array([[7, 8]])

    # create model
    gw = GridWorld(
        num_rows=num_rows,
        num_cols=num_cols,
        start_state=start_state,
        goal_states=goal_state,
    )
    gw.add_obstacles(obstacle_states=obstacles, bad_states=bad_states)
    gw.add_rewards(step_reward=-1, goal_reward=10, bad_state_reward=-6)
    gw.add_transition_probability(p_transition_success=0.7, bias=0.5)
    gw.add_discount(discount=0.9)
    model = gw.make()

    # solve with value iteration
    value_function, pi = policy_iteration(model, maxiter=100)

    # test value iteration outputs
    assert np.all(value_function == test_value)
    assert np.all(pi == test_pi)
