import math, random
class Layer:
    def __init__(self, num_of_nodes_in: int, num_of_nodes_out: int) -> None:
        self.num_of_nodes_in = num_of_nodes_in

        self.num_of_nodes_out = num_of_nodes_out

        self.weights = [[random.uniform(-1,1) for _ in range(self.num_of_nodes_in)] for _ in range(self.num_of_nodes_out)]

        self.biases = [0.0 for _ in range(num_of_nodes_out)]

        self.loss_gradient_w = [[0.0 for _ in range(self.num_of_nodes_in)] for _ in range(self.num_of_nodes_out)]

        self.loss_gradient_b = [0.0 for _ in range(self.num_of_nodes_out)]

    def calculate_outputs(self, inputs: list):
        weighted_inputs = [0.0 for _ in range(self.num_of_nodes_out)]

        for i in range(0, self.num_of_nodes_out, 1):
            weighted_input = self.biases[i]

            for j in range(0, self.num_of_nodes_in, 1):
                weighted_input += inputs[j] * self.weights[i][j]

            weighted_inputs[i] = self.activation(weighted_input)

        return weighted_inputs

    def activation(self, weighted_input):
        return (
            1
            / (1 + math.exp(-weighted_input))
            * (1 - 1 / (1 + math.exp(-weighted_input)))
        )

    def cost(self, output_activation, expected_output):
        return 2 * (output_activation - expected_output)

    def apply_gradient(self, learn_rate: float):
        for i in range(self.num_of_nodes_out):
            self.biases[i] -= self.loss_gradient_b[i] * learn_rate

            for j in range(self.num_of_nodes_in):
                self.weights[i][j] -= self.loss_gradient_w[i][j] * learn_rate

    def calculate_output_layer_node_values(self, expected_outputs):
        node_values = [0.0 for _ in range(len(expected_outputs))]
        
        for i in range(len(node_values)):
            loss_derivative = self.cost(activations[i], expected_outputs[i])
            
            activation = self.activation()
