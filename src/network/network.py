import numpy, json
from Layer import Layer


class Network:
    def __init__(self, model_path: str = "") -> None:
        self.layers = []

        self.file_path = model_path

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        output = X

        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict(self, input_data: list):
        x = numpy.array(input_data)

        return self.forward(x)

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

    def save_model(self, file_name: str, destination: str) -> bool:
        try:
            data = {}
            temp = []

            current_layer = 0
            for layer in self.layers:
                temp.append(
                    {
                        f"layer {current_layer}": {
                            "inputs": layer.input_size,
                            "outputs": layer.output_size,
                            "activation": layer.activation_type,
                            "weights": layer.weights.tolist(),
                            "bias": layer.bias.tolist(),
                        }
                    }
                )
                current_layer += 1

            data.update({"model": temp})

            save_path = f"{destination}/{file_name}.json"

            with open(save_path, "w") as file:
                json.dump(data, file, indent=4)
            return True
        except Exception as e:
            return False

    def load_model(self, path: str) -> bool:
        data = self._load_model(path)

        if not data or "model" not in data:
            print("Invalid or missing model data")
            return False

        self.layers.clear()

        for layer_dict in data["model"]:

            key = list(layer_dict.keys())[0]

            layer_info = layer_dict[key]

            layer = Layer(
                layer_info["inputs"], layer_info["outputs"], layer_info["activation"]
            )

            layer.weights = numpy.array(layer_info["weights"])

            layer.bias = numpy.array(layer_info["bias"])

            self.add_layer(layer)

        return True

    def _load_model(self, file_path: str) -> dict:
        if file_path == "":
            print(f"Empty File Path")

            return {}

        try:
            with open(file_path, "r") as file:

                return json.load(file)

        except Exception as e:
            print(f"Error loading model: {e}")

            return {}
