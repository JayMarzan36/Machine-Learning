import matplotlib.pyplot as plot


def plot_results(
    x,
    y_true,
    y_pred,
    title="Neural Network Approximation",
    xlabel="X",
    ylabel="Y",
    legend_labels=("True Function", "NN Prediction"),
):
    if len(y_true) <= 0 and len(y_pred) <= 0:
        plot.plot(x, label=legend_labels[0])
    else:
        plot.plot(x, y_true, label=legend_labels[0])
        plot.plot(x, y_pred, label=legend_labels[1])
    plot.legend()
    plot.title(title)
    plot.xlabel(xlabel)
    plot.ylabel(ylabel)
    plot.grid(True)
    plot.show()
