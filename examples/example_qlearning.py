import os
import timeit

import numpy as np

from algorithms.temporal_difference import QLearning
from env.grid_world import GridWorld
from policy import EpsilonGreedyPolicy
from utils.plots import plot_gridworld

###########################################################
#            Run Q-Learning on cliff walk                 #
###########################################################

# specify world parameters
num_rows = 3
num_cols = 3
start_state = np.array([[1, 1]])
obstacles = np.array([[2, 1], [0, 1]])
goal_states = np.array([[0, 0], [0, 2], [2, 2]])
goal_rewards = np.array([1, 10, -1000])
# create model
gw = GridWorld(
    num_rows=num_rows,
    num_cols=num_cols,
    start_state=start_state,
    goal_states=goal_states,
)

gw.add_obstacles(obstacle_states=obstacles)
gw.add_rewards(step_reward=0, goal_reward=goal_rewards, restart_state_reward=0)
gw.add_transition_probability(p_transition_success=1, bias=0)
gw.add_discount(discount=0.99)
model = gw.create_gridworld()
plot_gridworld(model, title="Test world")

rng = np.random.default_rng(0)

s = timeit.default_timer()

# solve with Q-Learning
solver = QLearning(
    model,
    behavior_policy=EpsilonGreedyPolicy(rng, epsilon=0.5),
    alpha=0.1,
    rng_state=rng,
)
solver.run(max_horizon=100, max_eps=10000)
q = solver.q_values
pi = solver.target_policy(q)
title = "Q-Learning"

# # solve with SARSA
# solver = Sarsa(model, alpha=0.1, epsilon=0.5)
# q, pi = solver.run(max_horizon=100, max_eps=10000)
# title = "SARSA"

print(q)
print(pi)
pi_arr = np.empty((num_rows, num_cols), dtype=object)
for i, policy in enumerate(pi):
    x, y = model.seq_to_state(i)
    pi_arr[x, y] = model.Action(np.argmax(policy)).name
print(pi_arr)

e = timeit.default_timer()
print("Time elapsed: ", e - s)

# plot the results
path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "doc",
    "imgs",
    "tiny_gridworld.png",
)
plot_gridworld(
    model, policy=pi, state_counts=solver.state_counts, title=title, path=path
)
