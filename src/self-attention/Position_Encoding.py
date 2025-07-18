import numpy
import json
import re

from Layer import Layer


def main(model_path: str, word_index_path: str, index_word_path: str, input: str):

    def load_word2vec_data(model_path: str, word_index_path: str, index_to_word_path: str):
        target_layer = 0

        full_model = None

        word_to_index = None

        index_to_word = None

        with open(model_path, "r") as file:
            full_model = json.load(file)

        model_layer = full_model["model"][target_layer]

        key = list(model_layer.keys())[0]

        layer_info = model_layer[key]

        layer = Layer(layer_info["inputs"], layer_info["outputs"])

        layer.weights = numpy.array(layer_info["weights"])

        layer.bias = numpy.array(layer_info["bias"])

        final_layer = layer

        with open(word_index_path, "r") as file:
            word_to_index = json.load(file)
            
        with open(index_to_word_path, "r") as file:
            index_to_word = json.load(file)
            
            index_to_word = {int(key): value for key, value in index_to_word.items()}
            

        return final_layer.weights, word_to_index, index_to_word

    embedding_matrix, word_to_index, index_to_word = load_word2vec_data(
        model_path, word_index_path, index_word_path
    )

    print("Loaded data")

    # Position embedding
    # This process can be calculated in parallel 

    def get_position_encoding(
        sequence_len: int, embedding_dim: int, n: int = 10_000
    ):
        P = numpy.zeros((sequence_len, embedding_dim))

        for i in range(sequence_len):
            for j in numpy.arange(int(embedding_dim / 2)):
                denominator = numpy.power(n, 2 * j / embedding_dim)

                P[i, 2 * j] = numpy.sin(i / denominator)
                P[i, 2 * j + 1] = numpy.cos(i / denominator)

        return P

    embedding_dim = 4

    token_indecies = []

    # Clean input 
    clean_sentence = input.lower()

    clean_sentence = re.sub(r"[^\w\s]", "", clean_sentence)

    tokens = clean_sentence.split()

    # Put vectors into an array
    sequence_vectors = []

    for i in tokens:
        sequence_vectors.append(embedding_matrix[word_to_index[i]])

    sequence_len = len(tokens)

    positions = get_position_encoding(sequence_len, embedding_dim)

    # Add position vectors to token vectors
    index_count = 0
    for i in positions:
        sequence_vectors[index_count] += i
        
        index_count += 1
        
    return sequence_vectors