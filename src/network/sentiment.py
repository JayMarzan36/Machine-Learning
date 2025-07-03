import numpy
import time
import matplotlib.pyplot as plot
import json

from Network import Network
from Layer import Layer
from util.token import Tokenizer
from util.plot import plot_results
from util.color import *

# 1,0,0,0 Positive
# 0,1,0,0 Negative
# 0,0,1,0 Neutral
# 0,0,0,1 Mixed

data = None
with open("C:/Users/jaymb/Downloads/sentiment_training_data.json") as file:
    data = json.load(file)


text = []
label = []

for item in data["training_data"]:
    text.append(item['text'])
    label.append(item['label'])

label = numpy.array(label)

# TODO vectorize the input text
tokenizer = Tokenizer("src/network/util/Vocab/vocab_4.json")

tokenized_input = tokenizer.batch_tokenize(text)

max_length = 0

tokenized_training_input = numpy.array([])

for i in range(len(tokenized_input)):
    current_length = len(tokenized_input[i])

    if current_length > max_length:
        max_length = current_length

temp = []
for j in range(len(tokenized_input)):
    current_array = numpy.array(tokenized_input[j])

    padded_array = numpy.pad(
        current_array,
        (0, max_length - len(current_array)),
        mode="constant",
        constant_values=0,
    )

    temp.append(padded_array)

tokenized_training_input = numpy.array(temp)

net = Network()

net.add_layer(Layer(max_length, 64, "lrelu"))
net.add_layer(Layer(64, 32, "lrelu"))
net.add_layer(Layer(32, 4, "softmax"))

epochs = 500_000

losses = net.train(
    tokenized_training_input,
    label,
    epochs=epochs,
    learning_rate=0.001,
    print_loss_every=int(epochs / 10),
)

prBold(f"Final Loss: {losses[len(losses) - 1]}")

plot_results(losses, [], [], legend_labels=("Loss"))
