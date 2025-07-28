import numpy, json, time
from Layer import Layer

from utilities import print_c, clear_screen, COLORS


class Network:
    def __init__(self, model_path: str = "") -> None:
        self.layers = []

        self.file_path = model_path

        self.input = None

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
        num_layers = len(self.layers)

        for i, layer in enumerate(reversed(self.layers)):
            layer_index = num_layers - 1 - i

            if layer_index == 0:
                A_prev = self.input
            else:
                A_prev = self.layers[layer_index - 1].output

            dZ = grad

            dW = A_prev.T @ dZ / dZ.shape[0]

            dB = numpy.sum(dZ, axis=0, keepdims=True) / dZ.shape[0]

            clip_value = 1.0

            dW = numpy.clip(dW, -clip_value, clip_value)

            dB = numpy.clip(dB, -clip_value, clip_value)

            self.adam_update(layer, dW, dB, learning_rate)

            # Only pass y_one_hot to the output (softmax) layer
            if hasattr(layer, "activation_type") and layer.activation_type == "softmax":

                grad = layer.backward(grad, self.last_y_batch, learning_rate)

            else:
                grad = layer.backward(grad, None, learning_rate)

    def train(
        self,
        X,
        y,
        epochs,
        learning_rate,
        early_stop: float = 0.01,
        model_name: str = "",
        save_path: str = "",
        save_model: bool = False,
        print_loss_every: int = 100,
    ):

        losses = []
        loss = 0
        previous_loss = numpy.inf

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

                self.input = X_batch

                output = self.forward(X_batch)  # Y_hat

                # Compute Loss (Cross Entropy)
                m = y_batch.shape[0]
                y_indices = numpy.argmax(y_batch, axis=1)
                log_likelihood = -numpy.log(output[range(m), y_indices])
                loss = numpy.sum(log_likelihood) / m
                losses.append(loss)

                if loss <= early_stop:
                    break

                # Compute gradient for softmax + cross-entropy
                grad = output - y_batch
                self.last_y_batch = y_batch  # Store for backward
                self.backward(grad, learning_rate)

            # Checking if print enable then printing every epoch % print_loss_every
            if print_loss_every != 0:
                if epoch % print_loss_every == 0:
                    print_c(f"Epoch {epoch}, Loss: {loss:.8}", "object_1")

        if save_model:
            status = self.save_model(model_name, save_path)

            if status:
                print_c(f"Saved {model_name} successfully", "success")
            else:
                print_c(f"Failed to save model", "error")

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

            with open(save_path, "w+") as file:
                json.dump(data, file, indent=4)
            return True
        except Exception as e:
            return False

    def load_model(self, path: str) -> bool:
        data = self._load_model(path)

        if not data or "model" not in data:
            print_c("Invalid or missing model data", "error")
            return False

        self.layers.clear()

        for layer_dict in data["model"]:

            key = list(layer_dict.keys())[0]

            layer_info = layer_dict[key]

            layer = Layer(layer_info["inputs"], layer_info["outputs"])

            layer.weights = numpy.array(layer_info["weights"])

            layer.bias = numpy.array(layer_info["bias"])

            self.add_layer(layer)

        return True

    def _load_model(self, file_path: str) -> dict:
        if file_path == "":
            print_c("Empty File Path", "error")

            return {}

        try:
            with open(file_path, "r") as file:

                return json.load(file)

        except Exception as e:
            print_c(f"Error loading mode: {e}", "error")

            return {}

    def get_batch_size(self, num_samples, fraction=0.1, min_size=4, max_size=128):
        batch_size = int(num_samples * fraction)

        batch_size = max(min_size, min(batch_size, max_size))

        return 2 ** int(round(numpy.log2(batch_size)))

    def adam_update(
        self, layer, dW, dB, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        layer.t += 1

        layer.m_w = beta1 * layer.m_w + (1 - beta1) * dW

        layer.m_b = beta1 * layer.m_b + (1 - beta1) * dB

        layer.v_w = beta2 * layer.v_w + (1 - beta2) * (dW**2)

        layer.v_b = beta2 * layer.v_b + (1 - beta2) * (dB**2)

        m_w_hat = layer.m_w / (1 - beta1**layer.t)

        m_b_hat = layer.m_b / (1 - beta1**layer.t)

        v_w_hat = layer.v_w / (1 - beta2**layer.t)

        v_b_hat = layer.v_b / (1 - beta2**layer.t)

        layer.weights -= learning_rate * m_w_hat / (numpy.sqrt(v_w_hat) + epsilon)
        layer.bias -= learning_rate * m_b_hat / (numpy.sqrt(v_b_hat) + epsilon)
