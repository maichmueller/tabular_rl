import sys

sys.path.append("..")
import numpy as np
from env.grid_world import GridWorld
from scipy.io import loadmat


def test_cliffworld():
    # load the test data
    grid_world = loadmat("../data/test_data/cliffworld.mat")["model"]

    # specify world parameters
    num_rows = 5
    num_cols = 10
    restart_states = np.array([[4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7]])
    obstacle_states = np.array([[0, 9], [1, 9], [2, 9], [3, 9], [4, 9]])
    start_state = np.array([[4, 0]])
    goal_states = np.array([[4, 8]])

    # create model
    gw = GridWorld(
        num_rows=num_rows,
        num_cols=num_cols,
        start_state=start_state,
        goal_states=goal_states,
    )
    gw.add_obstacles(
        obstacle_states=obstacle_states, restart_states=restart_states
    )
    gw.add_rewards(step_reward=-1, goal_reward=10, restart_state_reward=-100)
    gw.add_transition_probability(p_transition_success=1, bias=0)
    gw.add_discount(discount=0.9)
    model = gw.create_gridworld()

    # run tests
    assert np.all(model.reward == grid_world["R"][0][0][:, 0].reshape(-1, 1))
    assert np.all(model.probability[:, :, 0] == grid_world["P"][0][0][:, :, 0])
    assert np.all(model.probability[:, :, 1] == grid_world["P"][0][0][:, :, 1])
    assert np.all(model.probability[:, :, 2] == grid_world["P"][0][0][:, :, 2])
    assert np.all(model.probability[:, :, 3] == grid_world["P"][0][0][:, :, 3])
