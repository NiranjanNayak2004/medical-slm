import numpy as np


# -------------------------
# Input embeddings
# -------------------------

X = np.array([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0]
])


# -------------------------
# Learnable weight matrices
# -------------------------

W_Q = np.random.randn(3, 3)
W_K = np.random.randn(3, 3)
W_V = np.random.randn(3, 3)


# -------------------------
# Create Q, K, V
# -------------------------

Q = np.dot(X, W_Q)

K = np.dot(X, W_K)

V = np.dot(X, W_V)


print("Q:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)


# -------------------------
# Attention scores
# -------------------------

scores = np.dot(Q, K.T)

# Scale
scores = scores / np.sqrt(K.shape[1])


# -------------------------
# Softmax
# -------------------------

exp_scores = np.exp(
    scores - np.max(
        scores,
        axis=1,
        keepdims=True
    )
)

attention_weights = (
    exp_scores /
    np.sum(
        exp_scores,
        axis=1,
        keepdims=True
    )
)


print("\nAttention weights:")
print(attention_weights)


# -------------------------
# Attention output
# -------------------------

attention_output = np.dot(
    attention_weights,
    V
)

print("\nAttention output:")
print(attention_output)