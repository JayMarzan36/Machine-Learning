import numpy
import json

from util import pos_encoding, Network
from DecoderOnlyTransformer import DecoderOnlyTransformer

initialize_input = "looked up to him"

model_path = "src/word2vec/model/word2vec.json"
word_index_path = "src/word2vec/model/word2vec_word-to-index.json"
index_word_path = "src/word2vec/model/word2vec_index-to-word.json"

sequence_vectors = pos_encoding(
    model_path, word_index_path, index_word_path, initialize_input
)

embedding_dim = len(sequence_vectors[0])
transformer = DecoderOnlyTransformer(embedding_dim, num_heads=8, num_layers=6)

transformed_vectors = transformer.forward(sequence_vectors)

net = Network()
load = net.load_model(model_path)

if not load:
    raise Exception("Failed to load model")

with open(index_word_path, "r") as file:
    index_to_word = json.load(file)
    index_to_word = {int(key): value for key, value in index_to_word.items()}

output_layer = net.layers[1]

predictions = []
for i, vector in enumerate(transformed_vectors):
    probabilities = output_layer.forward(vector)
    highest_index = numpy.argmax(probabilities)
    highest_word = index_to_word[highest_index]
    predictions.append(highest_word)
    print(
        f"Position {i}: {highest_word} (probability: {probabilities[0][highest_index]:.4f})"
    )

print("\nFull sequence:", " ".join(predictions))
