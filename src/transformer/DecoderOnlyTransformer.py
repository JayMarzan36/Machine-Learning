import numpy
from util.LayerNorm import LayerNorm
from util.Layer import Layer
from util.Encoder_Layer import encoder
from util import pos_encoding


class DecoderOnlyTransformer:
    def __init__(self, embedding_dim, num_heads=8, num_layers=6):
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.layers = []
        self.cache = []

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

            attention_output = layer["attention"].self_attention_values

            residual = x + attention_output
            normed_attention = layer["ln1"].forward(residual)

            ffn1_output = layer["ffn1"].forward(normed_attention)

            ffn2_output = layer["ffn2"].forward(ffn1_output)

            residual = normed_attention + ffn2_output
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

            d_residual = layer["ln2"].backward(gradient, learning_rate)

            d_normed_attention = d_residual
            d_ffn = d_residual

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

        self.output_layer = Layer(self.embedding_dim, vocab_size, activation="softmax")

        for epoch in range(epochs):
            epoch_loss = 0

            for context, target in training_data:
                input_sequence = " ".join(context)
                sequence_vectors = pos_encoding(
                    model_path, word_index_path, index_word_path, input_sequence
                )

                transformed = self.forward(sequence_vectors)

                final_vector = transformed[-1].reshape(1, -1)

                target_vector = self.to_one_hot(target, word_to_index, vocab_size)
                target_vector = target_vector.reshape(1, -1)

                output = self.output_layer.forward(final_vector)

                epsilon = 1e-10
                loss = -numpy.sum(target_vector * numpy.log(output + epsilon))
                epoch_loss += loss

                output_gradient = output - target_vector

                gradient = self.output_layer.backward(
                    output_gradient, target_vector, learning_rate
                )

                self.backward(gradient, learning_rate)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

    def to_one_hot(self, word, word_to_index, vocab_size):
        """Convert word to one-hot vector"""
        vector = numpy.zeros(vocab_size)
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

            final_vector = transformed[-1].reshape(1, -1)

            logits = self.output_layer.forward(final_vector)

            logits = logits / temperature

            probs = numpy.exp(logits) / numpy.sum(numpy.exp(logits))

            next_word_idx = numpy.random.choice(len(word_to_index), p=probs.flatten())
            next_word = index_to_word[next_word_idx]

            current_sequence.append(next_word)

            if next_word == "<END>":
                break

        return " ".join(current_sequence)
