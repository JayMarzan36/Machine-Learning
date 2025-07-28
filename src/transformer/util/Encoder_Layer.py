import numpy
from .LayerNorm import LayerNorm


class encoder:
    def __init__(self, sequence_vectors, embedding_dim=None) -> None:
        if embedding_dim is None:
            embedding_dim = len(sequence_vectors[0])

        self.sequence_vectors = sequence_vectors
        self.embedding_dim = embedding_dim
        self.sequence_length = len(sequence_vectors)

        self.layer_norm = LayerNorm(self.embedding_dim)

        self.W_q = numpy.random.randn(self.embedding_dim, self.embedding_dim) * 0.1
        self.W_k = numpy.random.randn(self.embedding_dim, self.embedding_dim) * 0.1
        self.W_v = numpy.random.randn(self.embedding_dim, self.embedding_dim) * 0.1

        self.Q = numpy.dot(sequence_vectors, self.W_q)
        self.K = numpy.dot(sequence_vectors, self.W_k)
        self.V = numpy.dot(sequence_vectors, self.W_v)

        self.self_attention_values = []
        self.attention_weights = []
        self.masks = []

        for i in range(self.sequence_length):
            scores = numpy.dot(self.Q[i].reshape(1, -1), self.K.T) / numpy.sqrt(
                self.embedding_dim
            )
            scores = scores.reshape(-1)

            mask = numpy.ones(self.sequence_length)
            mask[i + 1 :] = 0
            self.masks.append(mask)

            scores = scores * mask + (1 - mask) * (-1e9)

            weights = self.softmax(scores)
            self.attention_weights.append(weights)
            attended = numpy.dot(weights, self.V)

            residual = attended + sequence_vectors[i]

            normalized = self.layer_norm.forward(residual)

            self.self_attention_values.append(normalized)

        self.self_attention_values = numpy.array(self.self_attention_values)
        self.attention_weights = numpy.array(self.attention_weights)

    def softmax(self, x):
        exp_x = numpy.exp(x - numpy.max(x))
        return exp_x / numpy.sum(exp_x)

    def backward(self, gradient, learning_rate):
        d_values = numpy.zeros_like(self.sequence_vectors)
        d_weights = {
            "q": numpy.zeros_like(self.W_q),
            "k": numpy.zeros_like(self.W_k),
            "v": numpy.zeros_like(self.W_v),
        }

        for i in range(self.sequence_length):
            d_residual = self.layer_norm.backward(gradient[i], learning_rate)

            d_attended = d_residual
            d_input = d_residual

            d_attended_2d = d_attended.reshape(1, -1)

            d_scores = numpy.zeros(self.sequence_length)

            mask = self.masks[i]

            attn_weights = self.attention_weights[i]
            softmax_grad = attn_weights * (1 - attn_weights)

            d_v_temp = numpy.outer(attn_weights, d_attended)
            d_weights["v"] += numpy.dot(
                self.sequence_vectors[i].reshape(-1, 1),
                numpy.mean(d_v_temp, axis=0).reshape(1, -1),
            )

            d_scores = softmax_grad * numpy.dot(d_attended, self.V.T)
            d_scores = d_scores * mask
            d_scores /= numpy.sqrt(self.embedding_dim)

            q_grad = numpy.dot(
                self.sequence_vectors[i].reshape(1, -1).T,
                numpy.dot(d_scores.reshape(1, -1), self.K),
            )
            k_grad = numpy.dot(
                self.sequence_vectors[i].reshape(1, -1).T,
                numpy.dot(d_scores.reshape(1, -1), self.Q),
            )

            d_weights["q"] += q_grad
            d_weights["k"] += k_grad

            d_values[i] = (
                numpy.dot(d_scores, self.K) @ self.W_q.T
                + numpy.dot(d_scores.T, self.Q) @ self.W_k.T
                + d_attended @ self.W_v.T
            )

        self.W_q -= learning_rate * d_weights["q"]
        self.W_k -= learning_rate * d_weights["k"]
        self.W_v -= learning_rate * d_weights["v"]

        return d_values
