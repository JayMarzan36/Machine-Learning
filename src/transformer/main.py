import json

from DecoderOnlyTransformer import DecoderOnlyTransformer

if __name__ == "__main__":
    model_path = "src/word2vec/model/word2vec.json"

    word_index_path = "src/word2vec/model/word2vec_word-to-index.json"

    index_word_path = "src/word2vec/model/word2vec_index-to-word.json"

    with open(word_index_path, "r") as file:
        word_to_index = json.load(file)

    with open(index_word_path, "r") as file:
        index_to_word = json.load(file)

    index_to_word = {int(key): value for key, value in index_to_word.items()}

    embedding_dim = 4

    save_path = "src/transformer/model/model.json"

    loaded_transformer = DecoderOnlyTransformer.load_model(
        save_path, vocab_size=len(word_to_index)
    )

    if loaded_transformer is not None:
        max_length = 25

        temperature = 0.7

        input_text = input("Input: ")

        valid_vocab = word_to_index.keys()

        splitted_input = input_text.split()

        valid_check = True

        for i in splitted_input:
            if i not in valid_vocab:
                print(f"Input {i} not in vocab")
                valid_vocab = False
                break

        if valid_check:
            print("")
            loaded_text = loaded_transformer.generate_sequence(
                seed_text=input_text,
                word_to_index=word_to_index,
                index_to_word=index_to_word,
                model_path="src/word2vec/model/word2vec.json",
                word_index_path="src/word2vec/model/word2vec_word-to-index.json",
                index_word_path="src/word2vec/model/word2vec_index-to-word.json",
                max_length=max_length,
                temperature=temperature,
            )
            loaded_text = loaded_transformer.generate_sequence(
                seed_text=loaded_text,
                word_to_index=word_to_index,
                index_to_word=index_to_word,
                model_path="src/word2vec/model/word2vec.json",
                word_index_path="src/word2vec/model/word2vec_word-to-index.json",
                index_word_path="src/word2vec/model/word2vec_index-to-word.json",
                max_length=max_length,
                temperature=temperature,
            )
            print("")
            print(loaded_text)