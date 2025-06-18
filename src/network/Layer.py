import numpy


class Layer:
    def __init__(self, input_size, output_size, activation="sigmoid") -> None:
        self.input_size = input_size

        self.output_size = output_size

        self.weights = numpy.random.randn(input_size, output_size) * numpy.sqrt(
            2.0 / input_size
        )

        self.bias = numpy.zeros((1, output_size))

        if activation:
            self._aply_activation(activation)
        else:
            # default if no activation found
            self.activation = self._linear

            self.activation_derivative = self._linear_derivative

            self.activation_type = "linear"

        self.input = None

        self.output = None

    def forward(self, input_data) -> float:
        self.input = input_data

        z = numpy.dot(input_data, self.weights) + self.bias

        self.output = self.activation(z)

        return self.output

    def backward(self, output_error, learning_rate):
        delta = output_error * self.activation_derivative(self.output)

        d_weights = numpy.dot(self.input.T, delta)

        d_bias = numpy.sum(delta, axis=0, keepdims=True)

        input_error = numpy.dot(delta, self.weights.T)

        self.weights -= learning_rate * d_weights

        self.bias -= learning_rate * d_bias

        return input_error

    def _aply_activation(self, type: str):
        if type == "sigmoid":
            self.activation = self._sigmoid

            self.activation_derivative = self._sigmoid_derivative

            self.activation_type = "sigmoid"

        elif type == "linear":
            self.activation = self._linear

            self.activation_derivative = self._linear_derivative

            self.activation_type = "linear"

        elif type == "lrelu":
            self.activation = self._leaky_relu

            self.activation_derivative = self._leaky_relu_derivative

            self.activation_type = "lrelu"

    def _sigmoid(self, x) -> float:
        return 1 / (1 + numpy.exp(-x))

    def _sigmoid_derivative(self, x) -> float:
        return x * (1 - x)

    def _linear(self, x) -> float:
        return x

    def _linear_derivative(self, x) -> float:
        return numpy.ones_like(x)

    def _leaky_relu(self, x, constant: float = 0.01) -> float:
        return numpy.maximum(x * constant, x)

    def _leaky_relu_derivative(self, x, constant: float = 0.01) -> float:
        dx = numpy.ones_like(x)
        dx[x < 0] = constant
        return dx
