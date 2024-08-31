import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_dist(
    samples,
    plot_Gaussian=True,
    fig=None,
    axes=None,
    plot_other_data=False,
    other_data=None,
):
    num_dimensions = samples.shape[1]
    num_rows = min(2, num_dimensions)
    num_cols = (num_dimensions + num_rows - 1) // num_rows

    if fig is None:
        fig = plt.figure(figsize=(8, 3 * num_rows))
    else:
        fig.clf()  # Clear current figure

    if axes is None:
        if num_rows == 1 and num_cols == 1:
            axes = fig.add_subplot(1, 1, 1)
        else:
            axes = fig.subplots(num_rows, num_cols)
            axes = axes.flatten()

    # Determine global min and max across all dimensions
    global_min = samples.min()
    global_max = samples.max()

    for i in range(num_dimensions):
        axes[i].hist(samples[:, i], bins=30, density=True, alpha=0.6)
        axes[i].set_title(f"Dim {i + 1}", fontsize=8)
        axes[i].tick_params(axis="both", which="both", labelsize=6)

        # Set uniform x and y axes
        axes[i].set_xlim(global_min, global_max)
        axes[i].set_ylim(0, 1.1 * axes[i].get_ylim()[1])

        if plot_Gaussian == True:
            axes[i].hist(torch.randn(1024), bins=30, density=True, alpha=0.1)
        if plot_other_data == True:
            axes[i].hist(other_data[:, i], bins=30, density=True, alpha=0.1)

    # Hide unused subplots
    for j in range(num_dimensions, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)


def plot_scatter(
    samples,
    plot_Gaussian=True,
    fig=None,
    axes=None,
    plot_other_data=False,
    other_data=None,
    loglog=False,
):
    num_dimensions = samples.shape[1]
    num_rows = min(2, num_dimensions)
    num_cols = (num_dimensions + num_rows - 1) // num_rows
    loglog = False
    if fig is None:
        fig = plt.figure(figsize=(8, 3 * num_rows))
    else:
        fig.clf()  # Clear current figure

    plt.scatter(samples[..., 0], samples[..., 1], alpha=0.5, label="Generated Points")
    if plot_other_data == True:
        plt.scatter(
            other_data[..., 0], other_data[..., 1], alpha=0.1, label="True Points"
        )
    if loglog:
        # Set the scales of the x and y axes to logarithmic
        plt.xscale("log")
        plt.yscale("log")
    plt.xlabel("$\log(x)$")
    plt.ylabel("$\log(Q^{2})$")
    plt.legend()
    plt.tight_layout()
    # plt.draw()
    # plt.pause(0.1)


def plot_params(
    means_over_epochs, std_devs_over_epochs, true_params, fig=None, axes=None
):
    # fig.clf()
    # param_names = ['$N_{u}$', '$a_{u}$', '$b_{u}$', '$N_{d}$', '$a_{d}$', '$b_{d}$']
    param_names = list(range(33))
    num_epochs = len(means_over_epochs)
    num_params = len(means_over_epochs[0])

    # Convert lists of lists to numpy arrays for easier handling
    means_over_epochs = np.array(means_over_epochs)
    std_devs_over_epochs = np.array(std_devs_over_epochs)

    # Determine the number of rows and columns for subplots
    nrows = (num_params + 2) // 3
    ncols = 3

    # Create figure and axes if not provided
    # if fig is None or axes is None:
    #     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 3))
    # Create axes if not provided
    fig.clf()
    if axes is None:
        axes = fig.subplots(nrows=nrows, ncols=ncols)
    # Ensure the number of subplots matches the number of parameters
    assert len(axes.flatten()) >= num_params, "Not enough subplots for all parameters."

    for i in range(num_params):
        ax = axes.flatten()[i]
        ax.errorbar(
            range(num_epochs),
            means_over_epochs[:, i],
            yerr=std_devs_over_epochs[:, i],
            fmt="-o",
            label=f"Parameter {i+1}",
        )
        ax.axhline(true_params[i], color="r", linestyle="--", label="True Value")
        ax.set_title(param_names[i])
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        # ax.legend()

    # Hide any empty subplots
    for j in range(num_params, len(axes.flatten())):
        axes.flatten()[j].set_visible(False)
    fig.tight_layout()

    # plt.tight_layout()
    # plt.draw()
    # plt.pause(0.1)
