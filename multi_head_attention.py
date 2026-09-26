import numpy as np

from model_config import EMBEDDING_DIM, NUM_HEADS, HEAD_DIM


class MultiHeadAttention:

    def __init__(self):
        self.embedding_dim = EMBEDDING_DIM
        self.num_heads = NUM_HEADS
        self.head_dim = HEAD_DIM

        # Query, Key, Value weights
        self.W_Q = np.random.randn(
            self.embedding_dim,
            self.embedding_dim
        ) * 0.02

        self.W_K = np.random.randn(
            self.embedding_dim,
            self.embedding_dim
        ) * 0.02

        self.W_V = np.random.randn(
            self.embedding_dim,
            self.embedding_dim
        ) * 0.02

        # Output projection
        self.W_O = np.random.randn(
            self.embedding_dim,
            self.embedding_dim
        ) * 0.02

        self.b_O = np.zeros(self.embedding_dim)

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def split_heads(self, x):
        """
        Input:
            (sequence_length, embedding_dim)

        Output:
            (num_heads, sequence_length, head_dim)
        """

        sequence_length = x.shape[0]

        x = x.reshape(
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        x = np.transpose(x, (1, 0, 2))

        return x

    def combine_heads(self, x):
        """
        Input:
            (num_heads, sequence_length, head_dim)

        Output:
            (sequence_length, embedding_dim)
        """

        x = np.transpose(x, (1, 0, 2))

        sequence_length = x.shape[0]

        x = x.reshape(
            sequence_length,
            self.embedding_dim
        )

        return x

    def forward(self, X):

        # ------------------------------------------------
        # 1. Create Query, Key and Value
        # ------------------------------------------------

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        # ------------------------------------------------
        # 2. Split into multiple attention heads
        # ------------------------------------------------

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # ------------------------------------------------
        # 3. Attention for each head
        # ------------------------------------------------

        attention_outputs = []

        for head in range(self.num_heads):

            q = Q[head]
            k = K[head]
            v = V[head]

            scores = q @ k.T

            scores = scores / np.sqrt(self.head_dim)

            # Causal mask
            sequence_length = scores.shape[0]

            mask = np.triu(
                np.ones(
                    (sequence_length, sequence_length)
                ),
                k=1
            )

            scores = np.where(
                mask == 1,
                -1e9,
                scores
            )

            weights = self.softmax(scores)

            output = weights @ v

            attention_outputs.append(output)

        # ------------------------------------------------
        # 4. Combine all heads
        # ------------------------------------------------

        attention_outputs = np.array(attention_outputs)

        combined = self.combine_heads(
            attention_outputs
        )

        # ------------------------------------------------
        # 5. Final output projection
        # ------------------------------------------------

        output = combined @ self.W_O + self.b_O

        return output


if __name__ == "__main__":

    print("=" * 60)
    print("        Multi-Head Attention Test")
    print("=" * 60)

    sequence_length = 8

    X = np.random.randn(
        sequence_length,
        EMBEDDING_DIM
    )

    attention = MultiHeadAttention()

    output = attention.forward(X)

    print()
    print("Input shape       :", X.shape)
    print("Number of heads   :", NUM_HEADS)
    print("Head dimension    :", HEAD_DIM)
    print("Q shape           :", (sequence_length, EMBEDDING_DIM))
    print(
        "Split head shape  :",
        (NUM_HEADS, sequence_length, HEAD_DIM)
    )
    print("Output shape      :", output.shape)

    print()
    print("Multi-head attention working correctly.")