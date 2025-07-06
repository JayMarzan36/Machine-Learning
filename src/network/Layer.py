import numpy


class Layer:
    def __init__(self, input_size, output_size, activation="sigmoid") -> None:
        self.input_size = input_size

        self.output_size = output_size

        self.weights = numpy.random.randn(input_size, output_size) * 0.1

        self.bias = numpy.zeros((1, output_size))

        self.m_w = numpy.zeros_like(self.weights)
        self.v_w = numpy.zeros_like(self.weights)
        self.m_b = numpy.zeros_like(self.bias)
        self.v_b = numpy.zeros_like(self.bias)
        self.t = 0

        if activation:
            self._apply_activation(activation)
        else:
            # default if no activation found
            self.activation = self._linear

            self.activation_derivative = self._linear_derivative

            self.activation_type = "linear"

        self.input = None

        self.output = None


    def forward(self, input_data) -> float:
        self.input = input_data

        # Softmax usage
        z = (
            numpy.dot(self.input, self.weights) + self.bias
        )

        a = self.activation(z)

        self.output = self._soft_max(a)

        return self.output

    def backward(self, grad, y_one_hot, learning_rate):
        if self.activation_type == "softmax":
            delta = grad * self.activation_derivative(self.output, y_one_hot)
        else:
            delta = grad * self.activation_derivative(self.output)

        d_weights = numpy.dot(self.input.T, delta)

        d_bias = numpy.sum(delta, axis=0, keepdims=True)

        input_error = numpy.dot(delta, self.weights.T)

        self.weights -= learning_rate * d_weights

        self.bias -= learning_rate * d_bias

        return input_error

    def _apply_activation(self, type: str):
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

        elif type == "softmax":
            self.activation = self._soft_max

            self.activation_derivative = self._soft_max_derivative

            self.activation_type = "softmax"

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + numpy.exp(-x))

    def _sigmoid_derivative(self, x: float) -> float:
        return x * (1 - x)

    def _linear(self, x: float) -> float:
        return x

    def _linear_derivative(self, x) -> float:
        return numpy.ones_like(x)

    def _leaky_relu(self, x: float, constant: float = 0.01) -> float:
        return numpy.maximum(x * constant, x)

    def _leaky_relu_derivative(self, x, constant: float = 0.01) -> float:
        dx = numpy.ones_like(x)
        dx[x < 0] = constant
        return dx

    def _soft_max(self, x) -> float:
        exp_logits = numpy.exp(x - numpy.max(x, axis=1, keepdims=True))

        return exp_logits / numpy.sum(exp_logits, axis=1, keepdims=True)

    def _soft_max_derivative(self, s, y):
        return numpy.subtract(s,y)
