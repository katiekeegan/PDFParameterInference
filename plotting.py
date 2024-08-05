import matplotlib.pyplot as plt
import torch


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
    # # Save the current frame as an image
    # buffer = io.BytesIO()
    # plt.savefig(buffer, format="png")
    # buffer.seek(0)
    # frames.append(Image.open(buffer))
