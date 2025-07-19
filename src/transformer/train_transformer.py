import numpy
import json
import re
from util import pos_encoding, Network
from DecoderOnlyTransformer import DecoderOnlyTransformer


def create_training_pairs(text, word_to_index, context_size=3):
    """Create training pairs for next word prediction"""
    words = text.lower()

    words = re.sub(r"[^\w\s]", "", words)
    words = words.split()
    pairs = []

    for i in range(len(words) - context_size):
        context = words[i : i + context_size]
        target = words[i + context_size]
        pairs.append((context, target))

    return pairs


model_path = "src/word2vec/model/word2vec.json"
word_index_path = "src/word2vec/model/word2vec_word-to-index.json"
index_word_path = "src/word2vec/model/word2vec_index-to-word.json"

with open(word_index_path, "r") as file:
    word_to_index = json.load(file)

with open(index_word_path, "r") as file:
    index_to_word = json.load(file)
    index_to_word = {int(key): value for key, value in index_to_word.items()}

embedding_dim = 4

transformer = DecoderOnlyTransformer(embedding_dim, num_heads=8, num_layers=6)


temp = []

with open("src/word2vec/long_text.txt", "r") as file:
    for line in file:
        temp.append(line)

final = ""

for i in temp:
    final += i


training_pairs = create_training_pairs(final, word_to_index)


transformer.train_transformer(
    training_pairs,
    word_to_index,
    index_to_word,
    model_path,
    word_index_path,
    index_word_path,
    epochs=200
)


def generate_next_word(
    transformer,
    input_text,
    word_to_index,
    index_to_word,
    model_path,
    word_index_path,
    index_word_path,
):
    sequence_vectors = pos_encoding(
        model_path, word_index_path, index_word_path, input_text
    )

    transformed = transformer.forward(sequence_vectors)

    final_vector = transformed[-1].reshape(1, -1)

    output = transformer.output_layer.forward(final_vector)

    highest_index = numpy.argmax(output)
    predicted_word = index_to_word[highest_index]

    return predicted_word


seed_text = "AI now"
generated_text = transformer.generate_sequence(
    seed_text=seed_text,
    word_to_index=word_to_index,
    index_to_word=index_to_word,
    model_path="src/word2vec/model/word2vec.json",
    word_index_path="src/word2vec/model/word2vec_word-to-index.json",
    index_word_path="src/word2vec/model/word2vec_index-to-word.json",
    max_length=20,
    temperature=0.7,
)
print(generated_text)
