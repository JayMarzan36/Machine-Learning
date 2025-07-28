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


model_path="src/word2vec/model/word2vec.json"
word_index_path="src/word2vec/model/word2vec_word-to-index.json"
index_word_path="src/word2vec/model/word2vec_index-to-word.json"

with open(word_index_path, "r") as file:
    word_to_index = json.load(file)

with open(index_word_path, "r") as file:
    index_to_word = json.load(file)
    index_to_word = {int(key): value for key, value in index_to_word.items()}

embedding_dim = 4

epochs = 100

transformer = DecoderOnlyTransformer(
    embedding_dim, vocab_size=len(word_to_index), num_heads=8, num_layers=6
)

temp = ""
count = 0
with open("src/word2vec/AllCombined.txt", "r", encoding="utf-8") as file:
    for line in file:
        if count >= 500:
            break
        temp += line
        count += 1


training_pairs = create_training_pairs(temp, word_to_index)

transformer.train_transformer(
    training_pairs,
    word_to_index,
    index_to_word,
    model_path,
    word_index_path,
    index_word_path,
    epochs=epochs,
)


transformer.save_model("src/transformer/model/model.json")
