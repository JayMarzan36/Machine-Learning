import numpy


class Query_Value():
    def __init__(self, embedding_dim: int, token_vector) -> None:
        self.vector = token_vector
        
        self.weights = numpy.random.randn(embedding_dim, embedding_dim) * 0.1
        
        self.value = []
        
        for weight in self.weights:
            self.value.append(numpy.sum(weight * self.vector))