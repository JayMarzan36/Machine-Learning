# Vectorize calculations for faster


import matplotlib.pyplot as plot



# X input
x_input = [
    0,
    1.11111111,
    2.22222222,
    3.33333333,
    4.44444444,
    5.55555556,
    6.66666667,
    7.77777778,
    8.88888889,
    10.00,
]


# Y output
actual = [
    2.49671415,
    4.92540915,
    8.04685818,
    12.08322457,
    14.40700672,
    18.59145718,
    21.13754108,
    25.07505072,
    28.64116851,
    32.31996742,
]

length_of_examples = len(x_input)

initial_weight = 0
initial_bias = 0


def compute_prediction(x, weight, bias):
    # Modal Function
    return weight * x + bias


def weight_slope(weight, bias):
    temp = 0
    for i in range(length_of_examples):
        temp += (
            (compute_prediction(x_input[i], weight, bias) - actual[i]) * 2 * x_input[i]
        )

    return temp / length_of_examples


def bias_slope(weight, bias):
    temp = 0
    for i in range(length_of_examples):
        temp += (compute_prediction(x_input[i], weight, bias) - actual[i]) * 2

    return temp / length_of_examples


def compute_loss(weight, bias):
    MSE_Loss = 0
    for i in range(length_of_examples):
        MSE_Loss += (actual[i] - compute_prediction(x_input[i], weight, bias)) ** 2

    MSE_Loss = MSE_Loss / length_of_examples

    return MSE_Loss


def calculate_gradient_descent(weight, bias, trials: int = 100):
    total_losses = []

    for i in range(trials):

        loss = compute_loss(weight, bias)

        total_losses.append(loss)

        weight -= (0.01 * weight_slope(weight, bias))

        bias -= (0.01 * bias_slope(weight, bias))




    # print(total_losses)
    print(f"Final weight : {weight}")
    print(f"Final bias : {bias}")
    print(f"Final Loss : {total_losses[trials-1]}")
    print(f"Loss reduction : {((total_losses[0]-total_losses[trials-1])/total_losses[0]) * 100}%")
    
    plot.plot(total_losses)
    plot.title("Loss over Epochs")
    plot.xlabel("Epoch")
    plot.ylabel("MSE Loss")
    plot.grid(True)
    
    plot.show()
    



calculate_gradient_descent(initial_weight, initial_bias, trials=1000)
