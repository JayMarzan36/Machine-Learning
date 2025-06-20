import numpy
import matplotlib.pyplot as plot
import time

from Network import network
from Layer import Layer


def train(
    epochs: int,
    save_model: bool = False,
    min_loss: float = 0.02,
    run_till_min: bool = False,
):

    # Generate data
    X = numpy.linspace(-2 * numpy.pi, 2 * numpy.pi, 100).reshape(-1, 1)
    y = numpy.sin(X)

    Z = []
    for i in X:
        Z.append(1 / (1 + numpy.exp(-i)))
    Z = numpy.array(Z)
    # Create network and add layers
    net = network()

    # 3 layers? 1-10-1
    net.add_layer(Layer(1, 10, activation="sigmoid"))
    net.add_layer(Layer(10, 1, activation="linear"))  # output layer, linear activation

    losses = []

    last_loss = [0, 0, 0]

    run_to_min_loss = []

    if run_till_min:
        start_time = time.time()

        epoch_inc = 5000

        stuck_limit = 5

        stuck_count = 0

        # Inital Train to set the initial loss
        losses = net.train(
            X,
            y,
            epochs=epochs,
            learning_rate=0.001,
            print_loss_every=0,
        )

        while losses[len(losses) - 1] >= min_loss:

            losses = net.train(
                X,
                y,
                epochs=epochs,
                learning_rate=0.001,
                print_loss_every=0,
            )

            if losses[len(losses) - 1] == last_loss[len(last_loss) - 1]:
                stuck_count += 1

            if stuck_count == stuck_limit:
                print(f"Hit stuck limit training stopped")
                break

            epochs += epoch_inc

            run_to_min_loss.append(losses[len(losses) - 1])

        print(f"Final Loss : {run_to_min_loss[len(run_to_min_loss) - 1]}")

        print(f"Train Time : {(time.time() - start_time)/60}")  # Minutes
        
        print(f"Final Epoch : {epochs}")

    else:

        start_time = time.time()

        losses = net.train(
            X, y, epochs=epochs, learning_rate=0.001, print_loss_every=int(epochs / 8)
        )

        print(f"Final Loss : {losses[len(losses) - 1]}")

        print(f"Train Time : {(time.time() - start_time)/60}")  # Minutes

    if save_model:
        status = net.save_model(
            "batch_test",
            "/home/jay/Documents/Machine-Learning/src/network/model",
        )

        if status:
            print(f"Model saved")
        else:
            print(f"Error saving model")
#TODO put the plotting into a function for clean look
    predictions = net.forward(X)
    # Plot results
    plot.plot(X, y, label="True Function")
    plot.plot(X, predictions, label="NN Prediction")
    plot.legend()
    plot.title("Neural Network Approximation")
    plot.grid(True)
    plot.show()

    if run_till_min:
        # # Plot loss curve
        plot.plot(run_to_min_loss)
    else:
        plot.plot(losses)

    plot.title("Loss Over Epochs")
    plot.xlabel("Epoch")
    plot.ylabel("MSE Loss")
    plot.grid(True)
    plot.show()


def load_and_predict():
    net = network()

    load = net.load_model(
        "D:/Projects/Code/Organized/Machine Learning/New folder/Machine-Learning/src/network/model/sin_model.json"
    )

    if load:
        input_data = [[1.57]]
        prediction = net.predict(input_data)

        print(prediction)


# load_and_predict()
train(epochs=1_000, save_model=True, run_till_min=True, min_loss=0.0002)





# Weird, the second log of data shows that it is more accurate with less time to train and lower Epoch

# Final Loss : 0.0006228365782891032
# Train Time : 7.51558168331782
# Final Epoch : 101000


# Final Loss : 0.00023845843572804852
# Train Time : 6.75608971118927
# Final Epoch : 96000

# Final Loss : 0.0001567144811065792
# Train Time : 8.309765632947286
# Final Epoch : 106000