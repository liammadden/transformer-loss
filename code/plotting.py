import os

import matplotlib.pyplot as plt

fontsize_axis = 7
fontsize_ticks = 6

default_height_to_width_ratio = (5.0**0.5 - 1.0) / 2.0


def plot_experiment(experiment, path):

    final_training_losses = []
    final_test_losses = []
    model_num_params = []
    for run in experiment.runs:
        final_training_losses.append(run.training_loss_values[-1])
        final_test_losses.append(run.test_loss_values[-1])
        model_num_params.append(run.model_num_params)

    # print(experiment.runs)

    nrows = 1
    ncols = 2
    _, ax = plot_settings(nrows=nrows, ncols=ncols)

    plot_lineplot(
        axis=ax[0],
        xdata=model_num_params,
        ydata=final_training_losses,
        xlabel="Number of Parameters",
        ylabel="Final Training Loss",
    )

    plot_lineplot(
        axis=ax[1],
        xdata=model_num_params,
        ydata=final_test_losses,
        xlabel="Number of Parameters",
        ylabel="Final Test Loss",
    )

    plt.savefig(
        os.path.join(path, "plots", "loss-plot" + ".pdf"),
        bbox_inches="tight",
    )
    return


def plot_lineplot(axis, xdata, ydata, xlabel, ylabel):
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.ticklabel_format(axis='x', style='sci', scilimits=(-4, 4))
    axis.plot(xdata, ydata)


def plot_settings(
    nrows=1, ncols=1, width=6.0, height_to_width_ratio=default_height_to_width_ratio
):
    subplot_width = width / ncols
    subplot_height = height_to_width_ratio * subplot_width
    height = subplot_height * nrows
    figsize = (width, height)

    plt.rcParams.update(
        {
            "axes.labelsize": fontsize_axis,
            "figure.figsize": figsize,
            "figure.constrained_layout.use": False,
            "figure.autolayout": False,
            "lines.linewidth": 2,
            "lines.marker": "o",
            "xtick.labelsize": fontsize_ticks,
            "ytick.labelsize": fontsize_ticks,
            "figure.dpi": 250,
        }
    )

    return plt.subplots(nrows, ncols, constrained_layout=True)
