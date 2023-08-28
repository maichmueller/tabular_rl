import numpy as np
from env.grid_world import GridWorld
from scipy.io import loadmat


def test_smallworld():
    small_world = loadmat("../data/test_data/smallworld.mat")["model"]

    # specify world parameters
    num_cols = 4
    num_rows = 4
    obstacles = np.array([[1, 1], [2, 1], [1, 2]])
    start_state = np.array([[0, 0]])
    goal_state = np.array([[3, 3]])

    # create model
    gw = GridWorld(
        num_rows=num_rows,
        num_cols=num_cols,
        start_states=start_state,
        goal_states=goal_state,
    )
    gw.add_obstacles(obstacle_states=obstacles)
    gw.add_rewards(step_reward=-1, goal_reward=10)
    gw.add_transition_probability(p_transition_success=0.8, bias=0.5)
    gw.add_discount(discount=0.9)
    model = gw.make()

    # run tests
    assert np.all(model._reward == small_world["R"][0][0][:, 0].reshape(-1, 1))
    assert np.all(model._probability[:, :, 0] == small_world["P"][0][0][:, :, 0])
    assert np.all(model._probability[:, :, 1] == small_world["P"][0][0][:, :, 1])
    assert np.all(model._probability[:, :, 2] == small_world["P"][0][0][:, :, 2])
    assert np.all(model._probability[:, :, 3] == small_world["P"][0][0][:, :, 3])
