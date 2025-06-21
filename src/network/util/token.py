import json

class Tokenizer:
    def __init__(self, vocab: str) -> None:

        self.vocab = self._load_vocab(vocab)

    def tokenize(self, input_data: str):
        data = input_data.lower().split(" ")

        tokenized = []

        for word in range(len(data)):

            if self.vocab["vocab"][data[word]]:

                tokenized.append(self.vocab["vocab"][data[word]])

            else:

                tokenized.append("UNK")

        return tokenized

    def reverse_tokenize(self, input_tokens: list):
        tokenized = []

        for token in input_tokens:

            token = str(token)

            if token in self.vocab["reverse"]:

                tokenized.append(self.vocab["reverse"][token])

            else:

                tokenized.append("UNK")

        return tokenized

    def _load_vocab(self, file_path: str):
        with open(file_path, "r") as file:

            return json.load(file)
