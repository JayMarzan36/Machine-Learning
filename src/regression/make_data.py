import random
from typing import Callable


# gen_data(10, [0, 10])
def gen_data(
    data_length: int, input_range: list[float], function: Callable[[float], float]
) -> dict:

    data = {"input": [], "output": []}

    for i in range(data_length):
        x = random.uniform(input_range[0], input_range[1])

        data["input"].append(x)

        y = function(x)

        data["output"].append(y)

    return data


def test(
    number_of_tests: int,
    input_range: list,
    modal_function: Callable[[float, float, float], float],
    weight: float,
    bias: float,
    function: Callable[[float], float],
) -> None:
    print("...Testing")
    accuracies = []
    for i in range(number_of_tests):
        x = random.uniform(input_range[0], input_range[1])

        y = modal_function(x, weight, bias)

        actual = function(x)

        accuracies.append(((1-(actual - y)) * 100))

    total = 0
    for j in range(len(accuracies)):
        total += accuracies[j]

    print(f"Average Accuracy : {total/len(accuracies)}")


def gen_multivariate_data(n, x_range, func):
    inputs = []
    outputs = []

    for _ in range(n):
        x1 = random.uniform(*x_range)
        x2 = random.uniform(*x_range)
        x3 = random.uniform(*x_range)

        x = [x1, x2, x3]
        y = func(x)

        inputs.append(x)
        outputs.append(y)

    return {"input": inputs, "output": outputs}


def test_multivariate(
    number_of_tests: int,
    input_range: list[float],
    modal_function: Callable[[list[float], list[float], float], float],
    weights: list[float],
    bias: float,
    true_function: Callable[[list[float]], float],
    num_features: int,
) -> None:
    print("...Testing")
    errors = []

    for _ in range(number_of_tests):
        # Generate a multivariate input vector
        x = [
            random.uniform(input_range[0], input_range[1]) for _ in range(num_features)
        ]

        predicted = modal_function(x, weights, bias)
        actual = true_function(x)

        error = abs(predicted - actual)
        errors.append(error)

    avg_error = sum(errors) / number_of_tests

    print(f"Average Absolute Error: {avg_error:.4f}")
