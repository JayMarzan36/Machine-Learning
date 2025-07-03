import json
import re

def main():
    # Replace this with your current vocab JSON file path
    json_file_path = "src/network/util/Vocab/vocab_2.json"

    # Sentences to process
    data = None
    with open("C:/Users/jaymb/Downloads/sentiment_training_data.json") as file:
        data = json.load(file)


    print(data["training_data"])

    text = []
    label = []

    for item in data["training_data"]:
        text.append(item["text"])
        label.append(item["label"])

    print(text)
    print(label)

    # Load existing vocab
    with open(json_file_path, "r") as f:
        vocab_data = json.load(f)

    vocab = vocab_data["vocab"]
    reverse = vocab_data["reverse"]
    next_index = max(map(int, reverse.keys())) + 1


    # Function to split punctuation and tokenize
    def tokenize(text):
        return re.findall(r"\w+|[^\w\s]", text)


    # Update vocab with new tokens
    for sentence in text:
        tokens = tokenize(sentence)
        for token in tokens:
            if token not in vocab:
                vocab[token] = next_index
                reverse[str(next_index)] = token
                next_index += 1

    # Save updated vocab
    with open("vocab_4.json", "w") as f:
        json.dump({"vocab": vocab, "reverse": reverse}, f, indent=4)

    print("Vocabulary updated and saved to vocab_updated.json.")

if __name__ == "__main__":
    main()