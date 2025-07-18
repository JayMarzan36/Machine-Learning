import numpy


class LayerNorm:
    def __init__(self, embedding_dim, epsilon=1e-6):
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.gamma = numpy.ones(embedding_dim)  # scale
        self.beta = numpy.zeros(embedding_dim)  # shift

    def forward(self, x):
        # Save input for backward pass
        self.x = x

        # Calculate mean and variance
        self.mean = numpy.mean(x, axis=-1, keepdims=True)
        self.var = numpy.var(x, axis=-1, keepdims=True)

        # Normalize
        self.std = numpy.sqrt(self.var + self.epsilon)
        self.normalized = (x - self.mean) / self.std

        # Scale and shift
        out = self.gamma * self.normalized + self.beta
        return out

    def backward(self, gradient, learning_rate=0.001):
        # Get batch size
        N = self.x.shape[0]

        # Gradients with respect to gamma and beta
        dgamma = numpy.sum(gradient * self.normalized, axis=0)
        dbeta = numpy.sum(gradient, axis=0)

        # Gradient with respect to normalized input
        dnormalized = gradient * self.gamma

        # Gradient with respect to input
        dx = (
            (1.0 / N)
            / self.std
            * (
                N * dnormalized
                - numpy.sum(dnormalized, axis=0)
                - self.normalized * numpy.sum(dnormalized * self.normalized, axis=0)
            )
        )

        # Update parameters
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta

        return dx
