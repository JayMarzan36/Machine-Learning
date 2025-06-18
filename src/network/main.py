import numpy
import matplotlib.pyplot as plot
import time

from Network import Network
from Layer import Layer


def train(epochs: int):
    start_time = time.time()

    # Generate data
    X = numpy.linspace(-2 * numpy.pi, 2 * numpy.pi, 100).reshape(-1, 1)
    y = numpy.sin(X)

    # Create network and add layers
    net = Network()

    # 3 layers? 1-10-1
    net.add_layer(Layer(1, 10, activation="sigmoid"))
    net.add_layer(Layer(10, 1, activation="linear"))  # output layer, linear activation

    # With current settings, takes ~6 min

    # For first layer
    # leaky relu 406 sec, final loss = 0.14962
    # sigmoid 392 sec, final loss = 0.0009977

    print_loss = int(epochs / 8)

    losses = net.train(
        X, y, epochs=epochs, learning_rate=0.001, print_loss_every=print_loss
    )

    print(f"Final Loss : {losses[len(losses) - 1]}")

    print(f"Train Time : {time.time() - start_time}")

    status = net.save_model(
        "sin_model",
        "D:/Projects/Code/Organized/Machine Learning/New folder/Machine-Learning/src/network/model",
    )

    if status:
        print(f"Model saved")
    else:
        print(f"Error saving model")

    predictions = net.forward(X)
    # Plot results
    plot.plot(X, y, label="True Function")
    plot.plot(X, predictions, label="NN Prediction")
    plot.legend()
    plot.title("Neural Network Approximation")
    plot.grid(True)
    plot.show()

    # # Plot loss curve
    # plot.plot(losses)
    # plot.title("Loss Over Epochs")
    # plot.xlabel("Epoch")
    # plot.ylabel("MSE Loss")
    # plot.grid(True)
    # plot.show()


def load_and_predict():
    net = Network()

    load = net.load_model(
        "D:/Projects/Code/Organized/Machine Learning/New folder/Machine-Learning/src/network/model/sin_model.json"
    )

    if load:
        input_data = [[1.57]]
        prediction = net.predict(input_data)

        print(prediction)


load_and_predict()
# train(epochs=10_000_000)
