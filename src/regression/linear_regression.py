# Vectorize calculations for faster speed


import matplotlib.pyplot as plot
import random, time, math

from make_data import gen_data, test


# Function for test data y ≈ 3.1 * x + 2


def data_function(x: float) -> float:
    return math.sin(x)


data = gen_data(10, [0, 50], data_function)

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

def show_data() -> None:
    plot.plot(data["input"], data["output"])
    plot.title("Data")
    plot.xlabel("Input")
    plot.ylabel("Output")
    plot.grid(True)

    plot.show()

from typing import Callable

def show_data_v_modal(
    weight: float, bias: float, modal_function: Callable[[float, float, float], float]
) -> None:
    plot.scatter(data["input"], data["output"])

    modal_output = []

    for i in data["input"]:
        modal_output.append(modal_function(i, weight, bias))

    plot.plot(data["input"], modal_output)

    plot.title("Data")
    plot.xlabel("Input")
    plot.ylabel("Output")
    plot.grid(True)

    plot.show()


def compute_prediction(x: float, weight: float, bias: float) -> float:
    # Modal Function
    return weight * x + bias


def weight_slope(weight: float, bias: float, batch: list) -> float:
    temp = 0
    for i in batch:
        temp += (
            (compute_prediction(data["input"][i], weight, bias) - data["output"][i])
            * 2
            * data["input"][i]
        )

    return temp / len(batch)


def bias_slope(weight: float, bias: float, batch: list) -> float:
    temp = 0
    for i in batch:
        temp += (
            compute_prediction(data["input"][i], weight, bias) - data["output"][i]
        ) * 2

    return temp / len(batch)


def compute_loss(weight: float, bias: float) -> float:
    MSE_Loss = 0
    for i in range(length_of_examples):
        MSE_Loss += (
            data["output"][i] - compute_prediction(data["input"][i], weight, bias)
        ) ** 2

    MSE_Loss = MSE_Loss / length_of_examples

    return MSE_Loss


def train(
    weight: float,
    bias: float,
    batch_size: int = 4,
    learning_rate: float = 0.01,
    trials: int = 100,
) -> list[float]:

    start_time = time.time()

    total_losses = []

    indicies = list(range(length_of_examples))

    for i in range(trials):

        random.shuffle(indicies)

        for start in range(0, length_of_examples, batch_size):
            batch_indicies = indicies[start : start + batch_size]

            weight_grad = weight_slope(weight, bias, batch_indicies)

            bias_grad = bias_slope(weight, bias, batch_indicies)

            weight -= learning_rate * weight_grad

            bias -= learning_rate * bias_grad

        loss = compute_loss(weight, bias)

        total_losses.append(loss)

        print(
            f"Epock {i}: Loss = {loss:.6f}, w = {weight:.4f}, b = {bias:.4f}", end="\r"
        )

    print("")

    final_time = time.time()

    print(f"... Finished Training in : {final_time - start_time}")

    return [weight, bias]


def main() -> None:
    final_weight, final_bias = train(
        initial_weight, initial_bias, learning_rate=0.001, trials=10000, batch_size=5
    )

    test(
        1000,
        [0, 50],
        compute_prediction,
        final_weight,
        final_bias,
        data_function
    )


    show_data_v_modal(final_weight, final_bias, compute_prediction)


if __name__ == "__main__":
    main()
