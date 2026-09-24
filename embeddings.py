import numpy as np


vocabulary = {
    "fever": 0,
    "headache": 1,
    "cough": 2,
    "cold": 3
}


embedding_size = 4

embedding_matrix = np.random.randn(
    len(vocabulary),
    embedding_size
)


def cosine_similarity(vector1, vector2):

    dot_product = np.dot(vector1, vector2)

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


word1 = "fever"
word2 = "headache"

vector1 = embedding_matrix[vocabulary[word1]]
vector2 = embedding_matrix[vocabulary[word2]]

similarity = cosine_similarity(vector1, vector2)


print("Word 1:", word1)
print("Vector 1:", vector1)

print("\nWord 2:", word2)
print("Vector 2:", vector2)

print("\nSimilarity:", similarity)