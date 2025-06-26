import numpy, json, time
from Layer import Layer


class network:
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

    def backward(self, grad, learning_rate):
        for i, layer in enumerate(reversed(self.layers)):
            # Only pass y_one_hot to the output (softmax) layer
            if hasattr(layer, 'activation_type') and layer.activation_type == "softmax":
                grad = layer.backward(grad, self.last_y_batch, learning_rate)
            else:
                grad = layer.backward(grad, None, learning_rate)

    def train(self, X, y, epochs, learning_rate, print_loss_every=100):
        losses = []
        loss = 0

        num_samples = X.shape[0]
        indices = numpy.arange(num_samples)

        # Set batch size
        batch_size = self.get_batch_size(num_samples)

        # Training loop
        for epoch in range(epochs):
            # Get start time for timing
            start_time = time.time()

            # Shuffle data at the start of each epoch
            numpy.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Batch training
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                output = self.forward(X_batch)  # Y_hat

                # Compute Loss (Cross Entropy)
                m = y_batch.shape[0]
                y_indices = numpy.argmax(y_batch, axis=1)
                log_likelihood = -numpy.log(output[range(m), y_indices])
                loss = numpy.sum(log_likelihood) / m
                losses.append(loss)

                # Compute gradient for softmax + cross-entropy
                grad = output - y_batch
                self.last_y_batch = y_batch  # Store for backward
                self.backward(grad, learning_rate)

            # Checking if print enable then printing every epoch % print_loss_every
            if print_loss_every != 0:
                if epoch % print_loss_every == 0:
                    print(
                        f"Epoch {epoch}, Loss: {loss:.4f}, Time:{time.time() - start_time}"
                    )

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
                layer_info["inputs"], layer_info["outputs"]
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

    def get_batch_size(self, num_samples, fraction=0.1, min_size=4, max_size=128): #TODO the min_size influences the 8 in (8,10) so the batch size
        batch_size = int(num_samples * fraction)

        batch_size = max(min_size, min(batch_size, max_size))

        return 2 ** int(round(numpy.log2(batch_size)))
