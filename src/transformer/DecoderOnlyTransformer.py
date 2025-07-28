import numpy
import json

from util.LayerNorm import LayerNorm
from util.Layer import Layer
from util.Encoder_Layer import encoder
from util import pos_encoding


class DecoderOnlyTransformer:
    def __init__(self, embedding_dim, vocab_size, num_heads=8, num_layers=6):
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.layers = []
        self.cache = []
        self.output_layer = Layer(self.embedding_dim, vocab_size, activation="softmax")

        for _ in range(num_layers):
            self.layers.append(
                {
                    "attention": None,
                    "ffn1": Layer(embedding_dim, embedding_dim * 4, activation="lrelu"),
                    "ffn2": Layer(embedding_dim * 4, embedding_dim, activation="lrelu"),
                    "ln1": LayerNorm(embedding_dim),
                    "ln2": LayerNorm(embedding_dim),
                }
            )

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer["attention"], type):
                layer["attention"] = layer["attention"](x)
            else:
                layer["attention"] = encoder(x)

            attention_output = layer["attention"].self_attention_values.astype(
                numpy.float32
            )

            residual = (x + attention_output).astype(numpy.float32)

            normed_attention = layer["ln1"].forward(residual)

            ffn1_output = layer["ffn1"].forward(normed_attention)

            ffn2_output = layer["ffn2"].forward(ffn1_output)

            residual = (normed_attention + ffn2_output).astype(numpy.float32)

            x = layer["ln2"].forward(residual)

            self.cache.append(
                {
                    "attention_output": attention_output,
                    "normed_attention": normed_attention,
                    "ffn_expanded": ffn1_output,
                    "ffn_output": ffn2_output,
                    "residual": residual,
                    "input": x,
                }
            )

        return x

    def backward(self, gradient, learning_rate):
        """Backward pass through the transformer"""
        for i in reversed(range(self.num_layers)):
            layer = self.layers[i]
            cache = self.cache[i]

            d_residual = layer["ln2"].backward(
                gradient.astype(numpy.float32), learning_rate
            )

            d_normed_attention = d_residual.astype(numpy.float32)

            d_ffn = d_residual.astype(numpy.float32)

            d_ffn = layer["ffn2"].backward(d_ffn, None, learning_rate)

            d_ffn = layer["ffn1"].backward(d_ffn, None, learning_rate)

            d_attention = layer["ln1"].backward(d_normed_attention, learning_rate)

            gradient = layer["attention"].backward(d_attention, learning_rate)

        self.cache = []

        return gradient

    def train_transformer(
        self,
        training_data,
        word_to_index,
        index_to_word,
        model_path,
        word_index_path,
        index_word_path,
        epochs=100,
        learning_rate=0.001,
    ):
        vocab_size = len(word_to_index)
        learning_rate = numpy.float32(learning_rate)

        for epoch in range(epochs):
            epoch_loss = numpy.float32(0)

            for context, target in training_data:
                input_sequence = " ".join(context)

                sequence_vectors = pos_encoding(
                    model_path, word_index_path, index_word_path, input_sequence
                )

                transformed = self.forward(sequence_vectors).astype(numpy.float32)

                final_vector = transformed[-1].reshape(1, -1).astype(numpy.float32)

                target_vector = self.to_one_hot(target, word_to_index, vocab_size)
                target_vector = target_vector.reshape(1, -1)

                output = self.output_layer.forward(final_vector)

                epsilon = numpy.float32(1e-10)
                loss = -numpy.sum(
                    target_vector * numpy.log(output + epsilon), dtype=numpy.float32
                )
                epoch_loss += loss

                output_gradient = (output - target_vector).astype(numpy.float32)

                gradient = self.output_layer.backward(
                    output_gradient, target_vector, learning_rate
                )

                self.backward(gradient, learning_rate)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

    def to_one_hot(self, word, word_to_index, vocab_size):
        """Convert word to one-hot vector"""
        vector = numpy.zeros(vocab_size, dtype=numpy.float32)
        vector[word_to_index[word]] = 1
        return vector

    def generate_sequence(
        self,
        seed_text,
        word_to_index,
        index_to_word,
        model_path,
        word_index_path,
        index_word_path,
        max_length=50,
        temperature=1.0,
    ):
        """
        Generate a sequence of text starting from a seed text.
        Args:
            seed_text (str): Initial text to start generation from
            word_to_index (dict): Mapping of words to indices
            index_to_word (dict): Mapping of indices to words
            model_path (str): Path to the word2vec model
            word_index_path (str): Path to word-to-index mapping
            index_word_path (str): Path to index-to-word mapping
            max_length (int): Maximum length of generated sequence
            temperature (float): Controls randomness in generation. Higher values = more random
        """
        current_sequence = seed_text.split()

        for _ in range(max_length):
            input_sequence = " ".join(current_sequence)
            sequence_vectors = pos_encoding(
                model_path, word_index_path, index_word_path, input_sequence
            )

            transformed = self.forward(sequence_vectors)

            final_vector = numpy.array(transformed[-1], dtype=numpy.float32).reshape(
                1, -1
            )

            logits = self.output_layer.forward(final_vector)

            logits = (logits / numpy.float32(temperature)).astype(numpy.float32)

            exp_logits = numpy.exp(logits).astype(numpy.float32)
            probs = exp_logits / numpy.sum(exp_logits, dtype=numpy.float32)

            next_word_idx = numpy.random.choice(len(word_to_index), p=probs.flatten())
            next_word = index_to_word[next_word_idx]

            current_sequence.append(next_word)

            if next_word == "<END>":
                break

        return " ".join(current_sequence)

    @classmethod
    def load_model(cls, load_path: str, vocab_size):
        """Load a transformer model from a JSON file.

        Args:
            load_path (str): Path to the saved model file

        Returns:
            DecoderOnlyTransformer: Loaded model instance
        """
        try:
            with open(load_path, "r") as file:
                data = json.load(file)

            layers = []

            loaded_layer_count = 0

            embedding_dim = data["model_config"].get("embedding_dim")
            num_layers = data["model_config"].get("num_layers")

            model = DecoderOnlyTransformer(
                embedding_dim=embedding_dim,
                vocab_size=vocab_size,
                num_layers=num_layers,
            )

            for i in range(num_layers):
                layer_keys = data["transformer_layers"][i].get("components").keys()

                temp = {}

                if "attention" not in layer_keys:
                    temp["attention"] = None

                for j in layer_keys:
                    current_layer_keys = (
                        data["transformer_layers"][i].get("components").get(j).keys()
                    )

                    if "gamma" in current_layer_keys:
                        current_layer = LayerNorm(
                            data["transformer_layers"][i]
                            .get("components")
                            .get(j)
                            .get("embedding_dim"),
                            data["transformer_layers"][i]
                            .get("components")
                            .get(j)
                            .get("epsilon"),
                        )

                        current_layer.gamma = numpy.array(
                            data["transformer_layers"][i]
                            .get("components")
                            .get(j)
                            .get("gamma"),
                        )
                        current_layer.beta = numpy.array(
                            data["transformer_layers"][i]
                            .get("components")
                            .get(j)
                            .get("beta"),
                        )

                        temp[j] = current_layer
                    elif "attention" == j:
                        current_layer = encoder(
                            numpy.zeros(
                                (
                                    1,
                                    data["transformer_layers"][i]
                                    .get("components")
                                    .get(j)
                                    .get("embedding_dim"),
                                ),
                                dtype=numpy.float32,
                            ),
                        )

                        current_layer.W_q = numpy.array(
                            data["transformer_layers"][i]
                            .get("components")
                            .get(j)
                            .get("W_q"),
                        )
                        current_layer.W_k = numpy.array(
                            data["transformer_layers"][i]
                            .get("components")
                            .get(j)
                            .get("W_k"),
                        )
                        current_layer.W_v = numpy.array(
                            data["transformer_layers"][i]
                            .get("components")
                            .get(j)
                            .get("W_v"),
                        )

                    else:
                        current_layer = Layer(
                            data["transformer_layers"][i]
                            .get("components")
                            .get(j)
                            .get("input_size"),
                            data["transformer_layers"][i]
                            .get("components")
                            .get(j)
                            .get("output_size"),
                        )

                        current_layer.weights = numpy.array(
                            data["transformer_layers"][i]
                            .get("components")
                            .get(j)
                            .get("weights"),
                        )
                        current_layer.bias = numpy.array(
                            data["transformer_layers"][i]
                            .get("components")
                            .get(j)
                            .get("bias"),
                        )

                        if j == "output_layer":
                            print("Loading output layer")
                            model.output_layer = current_layer

                        else:
                            temp[j] = current_layer

                loaded_layer_count += 1

            if loaded_layer_count != num_layers:
                raise Exception("Amount of layers wrong")

            model.layers = [dict((k, v) for k, v in layer.items()) for layer in layers]

            print(f"Model successfully loaded from {load_path}")

            return model

        except Exception as e:
            print(f"Error loading model: {str(e)}")

            return None

    def save_model(self, save_path: str):
        """Save the transformer model to a JSON file.

        Args:
            save_path (str): Path where to save the model

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            data = {
                "model_config": {
                    "embedding_dim": self.embedding_dim,
                    "num_layers": self.num_layers,
                }
            }

            transformer_layers = []
            for layer_idx, layer_dict in enumerate(self.layers):
                layer_data = {"layer_index": layer_idx, "components": {}}

                if layer_dict["attention"] is not None:
                    attention = layer_dict["attention"]
                    layer_data["components"]["attention"] = {
                        "embedding_dim": attention.embedding_dim,
                        "W_q": numpy.array(attention.W_q, dtype=numpy.float32).tolist(),
                        "W_k": numpy.array(attention.W_k, dtype=numpy.float32).tolist(),
                        "W_v": numpy.array(attention.W_v, dtype=numpy.float32).tolist(),
                    }

                for ffn_name in ["ffn1", "ffn2"]:
                    ffn = layer_dict[ffn_name]
                    layer_data["components"][ffn_name] = {
                        "input_size": ffn.input_size,
                        "output_size": ffn.output_size,
                        "weights": numpy.array(
                            ffn.weights, dtype=numpy.float32
                        ).tolist(),
                        "bias": numpy.array(ffn.bias, dtype=numpy.float32).tolist(),
                        "activation_type": ffn.activation_type,
                    }

                for ln_name in ["ln1", "ln2"]:
                    ln = layer_dict[ln_name]
                    layer_data["components"][ln_name] = {
                        "embedding_dim": ln.embedding_dim,
                        "epsilon": ln.epsilon,
                        "gamma": numpy.array(ln.gamma, dtype=numpy.float32)
                        .reshape(-1)
                        .tolist(),
                        "beta": numpy.array(ln.beta, dtype=numpy.float32)
                        .reshape(-1)
                        .tolist(),
                    }

                transformer_layers.append(layer_data)

            data["transformer_layers"] = transformer_layers

            if self.output_layer is not None:
                data["output_layer"] = {
                    "input_size": self.output_layer.input_size,
                    "output_size": self.output_layer.output_size,
                    "weights": numpy.array(
                        self.output_layer.weights, dtype=numpy.float32
                    ).tolist(),
                    "bias": numpy.array(
                        self.output_layer.bias, dtype=numpy.float32
                    ).tolist(),
                    "activation_type": self.output_layer.activation_type,
                }

            with open(save_path, "w") as file:
                json.dump(data, file, indent=4)

            print(f"Model successfully saved to {save_path}")
            return True

        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
