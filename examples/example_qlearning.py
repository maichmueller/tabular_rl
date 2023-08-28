import os
import timeit

import numpy as np

from algorithms import DoubleQLearning
from env.grid_world import GridWorld
from policy import EpsilonGreedyPolicyGenerator
from utils.plots import plot_gridworld

###########################################################
#            Run Q-Learning on tiny grid world            #
###########################################################

# specify world parameters
num_rows = 3
num_cols = 3
start_state = np.array([[1, 1]])
obstacles = np.array([[2, 1], [0, 1]])
goal_states = np.array([[0, 0], [0, 2], [2, 2]])
goal_rewards = np.array([1, 10, -1000])
# create model
model = GridWorld(
    num_rows=num_rows,
    num_cols=num_rows,
    start_states=start_state,
    goal_states=goal_states,
    goal_reward=goal_rewards,
    obstacle_states=obstacles,
    step_reward=0,
    restart_state_reward=0,
)

plot_gridworld(model, title="Test world")

rng = np.random.default_rng(0)

s = timeit.default_timer()

# solve with Q-Learning
solver = DoubleQLearning(
    model,
    behavior_policy=EpsilonGreedyPolicyGenerator(epsilon=0.5),
    alpha=0.1,
    discount=0.99,
    rng_state=rng,
)
title = "Q-Learning"

# # solve with SARSA
# solver = Sarsa(
#     model,
#     behavior_policy=EpsilonGreedyPolicy(epsilon=0.5),
#     alpha=0.1,
#     discount=0.99,
#     rng_state=rng,
# )
# title = "SARSA"

solver.run(max_horizon=100, max_eps=10000)
q = solver.q_values
pi = np.array([solver.target_policy(q_vals).probabilities for q_vals in q])

print(f"Final Q-Values:\n {q}")
print(f"Final Policy:\n {pi}")
pi_arr = np.empty((num_rows, num_cols), dtype=object)
for i, policy in enumerate(pi):
    x, y = model.index_to_state(i)
    pi_arr[x, y] = model.Action(np.argmax(policy)).name
with np.printoptions(formatter=dict(object=lambda string: f"{string:7s}")):
    print(f"Final Greedy Policy (Translated Names):\n {pi_arr}")


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
