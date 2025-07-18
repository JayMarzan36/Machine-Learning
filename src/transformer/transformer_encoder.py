from util import encoder, pos_encoding


input = "looked up to him"

model_path = "D:/Projects/Code/Organized/Machine Learning/New folder/Machine-Learning/src/word2vec/model/word2vec.json"
word_index_path = "D:/Projects/Code/Organized/Machine Learning/New folder/Machine-Learning/src/word2vec/model/word2vec_word-to-index.json"
index_word_path = "D:/Projects/Code/Organized/Machine Learning/New folder/Machine-Learning/src/word2vec/model/word2vec_index-to-word.json"

sequence_vectors = pos_encoding(
    model_path, word_index_path, index_word_path, input
)

layer = encoder(sequence_vectors)

print(layer.self_attention_values)