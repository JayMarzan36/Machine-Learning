import numpy


import Position_Encoding

input = "looked up to him"

model_path = "D:/Projects/Code/Organized/Machine Learning/New folder/Machine-Learning/src/self-attention/word2vec_data/word2vec.json"
word_index_path = "D:/Projects/Code/Organized/Machine Learning/New folder/Machine-Learning/src/self-attention/word2vec_data/word2vec_word-to-index.json"
index_word_path = "D:/Projects/Code/Organized/Machine Learning/New folder/Machine-Learning/src/self-attention/word2vec_data/word2vec_index-to-word.json"

sequence_vectors = Position_Encoding.main(
    model_path, word_index_path, index_word_path, input
)
embedding_dim = len(sequence_vectors[0])
sequence_length = len(sequence_vectors)


W_q = numpy.random.randn(embedding_dim, embedding_dim) * 0.1
W_k = numpy.random.randn(embedding_dim, embedding_dim) * 0.1
W_v = numpy.random.randn(embedding_dim, embedding_dim) * 0.1

Q = numpy.dot(sequence_vectors, W_q)  
K = numpy.dot(sequence_vectors, W_k)  
V = numpy.dot(sequence_vectors, W_v)  


def softmax(x):
    exp_x = numpy.exp(x - numpy.max(x))
    return exp_x / numpy.sum(exp_x)


self_attention_values = []
for i in range(sequence_length):
    scores = numpy.dot(Q[i], K.T) / numpy.sqrt(embedding_dim) 
    weights = softmax(scores)  
    attended = numpy.dot(weights, V)  
    self_attention_values.append(attended)

self_attention_values = numpy.array(self_attention_values)  
print(self_attention_values)
