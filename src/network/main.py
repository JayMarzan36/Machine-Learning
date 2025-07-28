import numpy
import matplotlib.pyplot as plot
import time

from Network import Network
from Layer import Layer
from util.token import Tokenizer
from util.plot import plot_results


def train(
    epochs: int,
    save_model: bool = False,
    min_loss: float = 0.02,
    run_till_min: bool = False,
):

    tokenizer = Tokenizer("src/network/util/Vocab/vocab_2.json")

    binary_logic_gates_in = numpy.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
        ]
    )
    # 1,0,0,0,0,0 AND
    # 0,1,0,0,0,0 OR
    # 0,0,1,0,0,0 NAND
    # 0,0,0,1,0,0 NOR
    # 0,0,0,0,1,0 EX-OR
    # 0,0,0,0,0,1 EX-NOR
    binary_logic_gates_out = numpy.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    binary_logic_gates_in = numpy.array(binary_logic_gates_in)

    # Create network and add layers
    net = Network()

    # 3 layers? 1-10-1
    net.add_layer(Layer(3, 16, activation="lrelu"))
    net.add_layer(Layer(16,8, activation="lrelu"))
    net.add_layer(Layer(8, 6, activation="softmax"))

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
            binary_logic_gates_in,
            binary_logic_gates_out,
            epochs=epochs,
            learning_rate=0.001,
            print_loss_every=0,
        )

        while losses[len(losses) - 1] >= min_loss:

            losses = net.train(
                binary_logic_gates_in,
                binary_logic_gates_out,
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
            binary_logic_gates_in,
            binary_logic_gates_out,
            epochs=epochs,
            learning_rate=0.01,
            print_loss_every=int(epochs / 8),
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

    predictions = net.forward(binary_logic_gates_in)

    test = net.forward([0, 0, 1])  # AND truth

    print(test)

    # 1,0,0,0,0,0 AND
    # 0,1,0,0,0,0 OR
    # 0,0,1,0,0,0 NAND
    # 0,0,0,1,0,0 NOR
    # 0,0,0,0,1,0 EX-OR
    # 0,0,0,0,0,1 EX-NOR

    #plot_results(binary_logic_gates_out, predictions, [])

    if run_till_min:
        # # Plot loss curve
        plot_results(run_to_min_loss, [], [], legend_labels=("Loss"))
    else:
        plot_results(losses, [], [], legend_labels=("Loss"))


def load_and_predict():
    net = Network()

    load = net.load_model(
        "D:/Projects/Code/Organized/Machine Learning/New folder/Machine-Learning/src/network/model/sin_model.json"
    )

    if load:
        input_data = [[1.57]]
        prediction = net.predict(input_data)

        print(prediction)


# load_and_predict()
train(epochs=100_000, save_model=False, run_till_min=True, min_loss=0.5)