import numpy
import time
import re
import math

from Network import Network
from Layer import Layer


# can calculate estimate loss with log(vocab_size)
training_sentence = "Once upon a time, in the grand kingdom of Everlight, a wise king and a brave queen ruled with kindness and strength. The king was known for his wisdom, and the queen was admired for her courage. Every man in the village respected the king for his fairness. Every woman looked up to the queen, who often walked among them, listening to their stories and helping those in need. One day, a poor man came to the castle gates. He asked to see the queen. She welcomed him kindly and gave him food and shelter. The king rewarded the woman who had guided him through the woods to the palace. As seasons passed, the king and queen continued to lead their people. The man who had once been hungry became a trusted advisor. The woman who helped him became the royal gardener, growing roses that the queen loved dearly. The king and queen ruled together for many years, and their names were remembered by every man, woman, and child in Everlight."

window_size = 2

embedding_dim = 10

# Process input sentence
training_sentence = training_sentence.lower()

training_sentence = re.sub(r"[^\w\s]", "", training_sentence)

tokens = training_sentence.split()

vocab = list(set(tokens))

print(f"Estimate loss {math.log(len(vocab))}")

# Setup input for array

word_to_index = {word : i for i, word in enumerate(vocab)}

index_to_word = {i: word for word, i in word_to_index.items()}

vocab_size = len(vocab)

def generate_skipgram_pairs(tokens: list, window_size: int = 2):
    pairs = []

    for i, center in enumerate(tokens):
        for j in range(i - window_size, i + window_size + 1):
            if j != i and 0 <= j < len(tokens):
                pairs.append((center, tokens[j]))
    return pairs

def to_one_hot(word, word_to_index, vocab_size):
    vector = numpy.zeros(vocab_size)

    vector[word_to_index[word]] = 1

    return vector

# Generate pairs

pairs = generate_skipgram_pairs(tokens)

# Make final array inputs for network

X = []
Y = []

for center_word, context_word in pairs:
    X.append(to_one_hot(center_word, word_to_index, vocab_size))
    Y.append(to_one_hot(context_word, word_to_index, vocab_size))

X = numpy.array(X)
Y = numpy.array(Y)

# Setup network

save_model = False
epochs = 500_000

net = Network()

net.add_layer(Layer(vocab_size, embedding_dim, activation="none"))
net.add_layer(Layer(embedding_dim, vocab_size, activation="softmax"))

# Train

start_time = time.time()

losses = net.train(
    X,
    Y,
    epochs=epochs,
    learning_rate=0.01,
    print_loss_every=int(epochs / 8),
)

print(f"Final Loss : {losses[len(losses) - 1]}")

print(f"Train Time : {(time.time() - start_time)/60}")  # Minutes

# Save model

if save_model:
    status = net.save_model(
        "word2vec",
        "/home/jay/Documents/Machine-Learning/src/word2vec/model",
)

    if status:
        print(f"Model saved")
    else:
        print(f"Error saving model")


# Function for getting similarity between vectors

def cosine_similarity(v1, v2):
    return numpy.dot(v1, v2) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2))


# Function for getting word

def vector_to_word(vector, embedding_matrix, index_to_word, top_n=1):
    similarities = []

    for i, vec in enumerate(embedding_matrix):
        sim = cosine_similarity(vector, vec)

        similarities.append((i, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return [index_to_word[i] for i, _ in similarities[:top_n]]


# Get embedding matrix from first layers weights
embedding_matrix = net.layers[0].weights

# Get vectors
king_vector = embedding_matrix[word_to_index["king"]]

man_vector = embedding_matrix[word_to_index["man"]]

queen_vector = embedding_matrix[word_to_index["queen"]]

woman_vector = embedding_matrix[word_to_index["woman"]]


# Test with king-man+woman=queen
result = king_vector - man_vector + woman_vector

result /= numpy.linalg.norm(result)


closest_word = vector_to_word(result, embedding_matrix, index_to_word, top_n=5)

print(closest_word)







