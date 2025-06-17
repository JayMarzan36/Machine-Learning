import numpy
from layer import Layer

class Network:
    def __init__(self) -> None:
        self.layers = []
        
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def forward(self, X):
        output = X
        
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, loss_grad, learning_rate):
        grad = loss_grad
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
            
    def train(self, X, y, epochs, learning_rate, print_loss_every=100):
        losses = []
        
        for epoch in range(epochs):
            output = self.forward(X)
            
            loss = numpy.mean((output - y) ** 2)
            
            losses.append(loss)
            
            loss_grad = 2 * (output - y) / y.size
            
            self.backward(loss_grad, learning_rate)
            
            if epoch % print_loss_every == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses