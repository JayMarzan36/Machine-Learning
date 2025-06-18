# Vectorize calculations for faster speed


import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random, time
from typing import Callable, Optional

from make_data import gen_multivariate_data, test_multivariate


# Function for test data y ≈ 3.1 * x + 2


def data_function(x: list[float]) -> float:
    return 2.5 * x[0] - 1.2 * x[1] + 0.7  # some weights and a bias


data = gen_multivariate_data(10, [0, 5], data_function)

length_of_examples = len(data["input"])

initial_weight = 0

initial_bias = 0


def save(final_weight: float, final_bias: float, file_name: str) -> bool:
    try:
        with open(file_name, "w") as file:

            file.write(f"weight : {final_weight}\nbias : {final_bias}")

    except Exception as e:
        print("...Could not save")
        return False

    return True


def show_loss(total_losses) -> None:
    plot.plot(total_losses)
    plot.title("Loss over Epochs")
    plot.xlabel("Epoch")
    plot.ylabel("MSE Loss")
    plot.grid(True)

    plot.show()


def show_data(data: dict, feature_index: int = 0) -> None:
    x_values = [x[feature_index] for x in data["input"]]
    y_values = data["output"]

    plot.scatter(x_values, y_values)
    plot.title(f"Data (Feature {feature_index} vs Output)")
    plot.xlabel(f"Feature {feature_index}")
    plot.ylabel("Output")
    plot.grid(True)
    plot.show()


def show_data_v_modal(
    weights: list[float],
    bias: float,
    modal_function: Callable[[list[float], list[float], float], float],
    data: dict,
    feature_index: int = 0,
) -> None:
    # Use only one feature (e.g., x[0]) for x-axis
    x_values = [x[feature_index] for x in data["input"]]
    actual_outputs = data["output"]

    predicted_outputs = [modal_function(x, weights, bias) for x in data["input"]]

    plot.scatter(x_values, actual_outputs, label="Actual")
    plot.plot(x_values, predicted_outputs, color="red", label="Predicted")

    plot.title(f"Model Fit (Feature {feature_index} vs Output)")
    plot.xlabel(f"Feature {feature_index}")
    plot.ylabel("Output")
    plot.legend()
    plot.grid(True)
    plot.show()


def show_heatmap(
    weights: list[float],
    bias: float,
    modal_function: Callable[[list[float], list[float], float], float],
    feature_x_index: int,
    feature_y_index: int,
    input_range: list[float],
    resolution: int = 100,
    fixed_values: Optional[list[float]] = None,
) -> None:
    # Create grid of x and y values
    x_vals = np.linspace(input_range[0], input_range[1], resolution)
    y_vals = np.linspace(input_range[0], input_range[1], resolution)
    z_vals = np.zeros((resolution, resolution))

    # Default fixed values for remaining features
    if fixed_values is None:
        fixed_values = [0.0] * len(weights)

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            input_vector = fixed_values.copy()
            input_vector[feature_x_index] = x
            input_vector[feature_y_index] = y
            z_vals[j, i] = modal_function(input_vector, weights, bias)

    plot.figure(figsize=(8, 6))
    plot.imshow(
        z_vals,
        extent=(input_range[0], input_range[1], input_range[0], input_range[1]),
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    plot.colorbar(label="Model Output")
    plot.title(
        f"Model Output Heatmap\nFeature {feature_x_index} vs Feature {feature_y_index}"
    )
    plot.xlabel(f"Feature {feature_x_index}")
    plot.ylabel(f"Feature {feature_y_index}")
    plot.grid(False)
    plot.show()


def show_3d_plot(
    weights: list[float],
    bias: float,
    modal_function: Callable[[list[float], list[float], float], float],
    feature_x_index: int,
    feature_y_index: int,
    input_range: list[float],
    resolution: int = 50,
    fixed_values: Optional[list[float]] = None,
) -> None:
    # Create a grid of X and Y values
    x_vals = np.linspace(input_range[0], input_range[1], resolution)
    y_vals = np.linspace(input_range[0], input_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    # Default fixed values for non-plotted features
    if fixed_values is None:
        fixed_values = [0.0] * len(weights)

    # Compute predictions
    for i in range(resolution):
        for j in range(resolution):
            input_vector = fixed_values.copy()
            input_vector[feature_x_index] = X[i, j]
            input_vector[feature_y_index] = Y[i, j]
            Z[i, j] = modal_function(input_vector, weights, bias)

    # Plotting
    fig = plot.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.8)
    ax.set_title(
        f"3D Model Output Surface\nFeature {feature_x_index} vs Feature {feature_y_index}"
    )
    ax.set_xlabel(f"Feature {feature_x_index}")
    ax.set_ylabel(f"Feature {feature_y_index}")
    ax.set_zlabel("Predicted Output")
    plot.show()


def compute_prediction(x: list[float], weights: list[float], bias: float) -> float:
    return sum(w * xi for w, xi in zip(weights, x)) + bias


def weight_slope(
    weights: list[float],
    bias: float,
    batch: list,
    inputs: list[list[float]],
    outputs: list[float],
) -> list[float]:
    n_features = len(weights)
    gradients = [0.0 for _ in range(n_features)]

    for i in batch:
        prediction = compute_prediction(inputs[i], weights, bias)
        error = prediction - outputs[i]
        for j in range(n_features):
            gradients[j] += 2 * error * inputs[i][j]

    return [g / len(batch) for g in gradients]


def bias_slope(
    weights: list[float],
    bias: float,
    batch: list,
    inputs: list[list[float]],
    outputs: list[float],
) -> float:
    temp = 0.0
    for i in batch:
        prediction = compute_prediction(inputs[i], weights, bias)
        temp += 2 * (prediction - outputs[i])
    return temp / len(batch)


def compute_loss(
    weights: list[float], bias: float, inputs: list[list[float]], outputs: list[float]
) -> float:
    total = 0
    for x, y in zip(inputs, outputs):
        total += (compute_prediction(x, weights, bias) - y) ** 2
    return total / len(inputs)


def train(
    weights: list[float],
    bias: float,
    inputs: list[list[float]],
    outputs: list[float],
    batch_size: int = 4,
    learning_rate: float = 0.01,
    trials: int = 100,
) -> tuple[list[float], float]:

    start_time = time.time()

    total_losses = []

    indices = list(range(len(inputs)))

    for epoch in range(trials):
        random.shuffle(indices)
        for start in range(0, len(inputs), batch_size):
            batch = indices[start : start + batch_size]

            grad_w = weight_slope(weights, bias, batch, inputs, outputs)
            grad_b = bias_slope(weights, bias, batch, inputs, outputs)

            # Gradient descent update
            weights = [w - learning_rate * gw for w, gw in zip(weights, grad_w)]
            bias -= learning_rate * grad_b

        # Optional: compute loss
        loss = compute_loss(weights, bias, inputs, outputs)
        total_losses.append(loss)

    print(f"Train time : {time.time() - start_time}")

    return weights, bias


def main() -> None:
    final_weight, final_bias = train(
        [0.0, 0.0],
        initial_bias,
        data["input"],
        data["output"],
        learning_rate=0.001,
        trials=10000,
        batch_size=5,
    )

    test_multivariate(
        1000, [0, 50], compute_prediction, final_weight, final_bias, data_function, 2
    )


if __name__ == "__main__":
    main()
