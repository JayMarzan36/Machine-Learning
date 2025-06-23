import json, math

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

    def reverse_tokenize(self, input_tokens: list | int | float):

        if type(input_tokens) == list:
            tokenized = []

            for token in input_tokens:

                token = str(token)

                if token in self.vocab["reverse"]:

                    tokenized.append(int(self.vocab["reverse"][token]))

                else:

                    tokenized.append("UNK")

            return tokenized

        elif type(input_tokens) == int:

            return [int(self.vocab["reverse"][input_tokens])]

        elif type(input_tokens) == float:

            return [int(self.vocab["reverse"][str(round(input_tokens))])]

    def _load_vocab(self, file_path: str):
        with open(file_path, "r") as file:

            return json.load(file)
