import itertools

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from utils.helper_functions import create_policy_direction_arrays
import numpy as np


def plot_gridworld(
    model, value_function=None, policy=None, state_counts=None, title=None, path=None
):
    """
    Plots the grid world solution.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    value_function : numpy array of shape (N, 1)
        Value function of the environment where N is the number
        of states in the environment.

    policy : numpy array of shape (N, 1)
        Optimal policy of the environment.

    title : string
        Title of the plot. Defaults to None.

    path : string
        Path to save image. Defaults to None.
    """

    if value_function is not None and state_counts is not None:
        raise Exception("Must supple either value function or state_counts, not both!")

    fig, ax = plt.subplots()

    # add features to grid world
    if value_function is not None:
        add_value_function(model, value_function, "Value function")
    elif state_counts is not None:
        add_value_function(model, state_counts, "State counts")
    elif value_function is None and state_counts is None:
        add_value_function(model, value_function, "Value function")

    add_patches(model, ax)
    add_policy(model, policy)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fancybox=True,
        shadow=True,
        ncol=3,
    )
    yticks = np.arange(-0.5, model.num_rows - 0.5, step=0.5)
    xticks = np.arange(-0.5, model.num_cols - 0.5, step=0.5)
    plt.yticks(yticks, labels=yticks.astype(int))
    plt.xticks(xticks, labels=xticks.astype(int))
    plt.gca().xaxis.tick_top()
    for label in plt.gca().xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
        plt.gca().xaxis

    for label in plt.gca().yaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    for label in plt.gca().get_ygridlines()[1::2]:
        label.set_visible(False)
        plt.gca().xaxis

    for label in plt.gca().get_xgridlines()[1::2]:
        label.set_visible(False)

    if title is not None:
        plt.title(title, fontdict=None, loc="center")
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()


def add_value_function(model, value_function, name):
    if value_function is not None:
        # colobar max and min
        vmin = np.min(value_function)
        vmax = np.max(value_function)
        cmap = plt.cm.get_cmap("cividis")
        if name == "State counts":
            vmin = 0
            # use second highest value as max (because starting tile will be highest)
            vmax = np.max(value_function[np.where(value_function != vmax)])
        # reshape and set obstacle states to low value
        val = value_function[:, 0].reshape(model.num_rows, model.num_cols)
        if model.obs_states is not None:
            index = model.obs_states
            val[index[:, 0], index[:, 1]] = vmin
        plt.imshow(val, vmin=vmin, vmax=vmax, zorder=0, cmap=cmap)
        plt.colorbar(label=name)
        if name == "State counts":
            # Create a rectangle patch for the start state to make it stand out
            rectangle = patches.Rectangle(
                model.start_states[0] - np.array([0.5, 0.505]),
                1,
                1,
                linewidth=0.1,
                edgecolor="black",
                facecolor="cyan",
            )

            # Add the rectangle to the plot
            plt.gca().add_patch(rectangle)
    else:
        val = np.zeros((model.num_rows, model.num_cols))
        plt.imshow(val, zorder=0, cmap=plt.cm.get_cmap("cividis"))
        plt.yticks(np.arange(-0.5, model.num_rows + 0.5, step=1))
        plt.xticks(np.arange(-0.5, model.num_cols + 0.5, step=1))
        plt.grid()
        plt.colorbar(label=name)


def add_patches(model, ax):
    start = patches.Circle(
        tuple(np.flip(model.start_states[0])),
        0.2,
        linewidth=1,
        edgecolor="b",
        facecolor="b",
        zorder=1,
        label="Start",
    )
    ax.add_patch(start)

    for i in range(model.goal_states.shape[0]):
        location = tuple(np.flip(model.goal_states[i, :]))
        end = patches.RegularPolygon(
            location,
            numVertices=5,
            radius=0.25,
            orientation=np.pi,
            edgecolor="g",
            zorder=1,
            facecolor="g",
            label="Goal" if i == 0 else None,
        )
        reward = model.reward(model.goal_states[i])
        reward_str = (
            f"{reward:.1f}"
            if not np.isclose(reward - int(reward), 0.0)
            else f"{reward:.0f}"
        )
        ax.text(*location, reward_str, ha="center", va="center")
        ax.add_patch(end)

    # obstacle states patches
    if model.obs_states is not None:
        for i in range(model.obs_states.shape[0]):
            obstacle = patches.Rectangle(
                tuple(np.flip(model.obs_states[i, :]) - 0.35),
                0.7,
                0.7,
                linewidth=1,
                edgecolor="black",
                facecolor="black",
                zorder=1,
                label="Obstacle" if i == 0 else None,
            )
            ax.add_patch(obstacle)

    if model.bad_states is not None:
        for i in range(model.bad_states.shape[0]):
            bad = patches.Wedge(
                tuple(np.flip(model.bad_states[i, :])),
                0.2,
                40,
                -40,
                linewidth=1,
                edgecolor="r",
                facecolor="r",
                zorder=1,
                label="Bad state" if i == 0 else None,
            )
            ax.add_patch(bad)

    if model.restart_states is not None:
        for i in range(model.restart_states.shape[0]):
            restart = patches.Wedge(
                tuple(np.flip(model.restart_states[i, :])),
                0.2,
                40,
                -40,
                linewidth=1,
                edgecolor="y",
                facecolor="y",
                zorder=1,
                label="Restart state" if i == 0 else None,
            )
            ax.add_patch(restart)


def add_policy(model, policy):
    if policy is not None:
        # define the gridworld
        X = np.arange(0, model.num_cols, 1)
        Y = np.arange(0, model.num_rows, 1)

        # define the policy direction arrows
        U, V = create_policy_direction_arrays(model, policy)
        # remove the obstacles and final state arrows
        ra = model.goal_states
        U[ra[:, 0], ra[:, 1]] = np.nan
        V[ra[:, 0], ra[:, 1]] = np.nan
        if model.obs_states is not None:
            ra = model.obs_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan
        if model.restart_states is not None:
            ra = model.restart_states
            U[ra[:, 0], ra[:, 1]] = np.nan
            V[ra[:, 0], ra[:, 1]] = np.nan

        plt.quiver(X, Y, U, V, zorder=10, label="Policy", color="orange")
