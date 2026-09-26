import numpy as np

from model_config import EMBEDDING_DIM, NUM_HEADS, HEAD_DIM


class MultiHeadAttention:

    def __init__(self):

        self.embedding_dim = EMBEDDING_DIM
        self.num_heads = NUM_HEADS
        self.head_dim = HEAD_DIM

        # Forward parameters
        self.W_Q = np.random.randn(
            EMBEDDING_DIM,
            EMBEDDING_DIM
        ) * 0.02

        self.W_K = np.random.randn(
            EMBEDDING_DIM,
            EMBEDDING_DIM
        ) * 0.02

        self.W_V = np.random.randn(
            EMBEDDING_DIM,
            EMBEDDING_DIM
        ) * 0.02

        self.W_O = np.random.randn(
            EMBEDDING_DIM,
            EMBEDDING_DIM
        ) * 0.02

        self.b_O = np.zeros(
            EMBEDDING_DIM
        )

        # Cache used during backward
        self.cache = None


    def softmax(self, x):

        x = x - np.max(
            x,
            axis=-1,
            keepdims=True
        )

        exp_x = np.exp(x)

        return exp_x / np.sum(
            exp_x,
            axis=-1,
            keepdims=True
        )


    def split_heads(self, x):

        sequence_length = x.shape[0]

        x = x.reshape(
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        return np.transpose(
            x,
            (1, 0, 2)
        )


    def combine_heads(self, x):

        x = np.transpose(
            x,
            (1, 0, 2)
        )

        sequence_length = x.shape[0]

        return x.reshape(
            sequence_length,
            self.embedding_dim
        )


    def forward(self, X):

        # -----------------------------------------
        # Linear projections
        # -----------------------------------------

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        # -----------------------------------------
        # Split into attention heads
        # -----------------------------------------

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_outputs = []
        attention_weights = []
        attention_scores = []

        # -----------------------------------------
        # Attention for each head
        # -----------------------------------------

        for head in range(self.num_heads):

            q = Q[head]
            k = K[head]
            v = V[head]

            scores = q @ k.T

            scores = (
                scores /
                np.sqrt(self.head_dim)
            )

            sequence_length = scores.shape[0]

            # Causal mask
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

            weights = self.softmax(
                scores
            )

            output = weights @ v

            attention_scores.append(scores)
            attention_weights.append(weights)
            attention_outputs.append(output)

        attention_outputs = np.array(
            attention_outputs
        )

        attention_weights = np.array(
            attention_weights
        )

        attention_scores = np.array(
            attention_scores
        )

        # -----------------------------------------
        # Combine heads
        # -----------------------------------------

        combined = self.combine_heads(
            attention_outputs
        )

        output = (
            combined @ self.W_O
            + self.b_O
        )

        # Save values for backward
        self.cache = {
            "X": X,
            "Q": Q,
            "K": K,
            "V": V,
            "weights": attention_weights,
            "outputs": attention_outputs,
            "combined": combined
        }

        return output


    def backward(self, d_output):

        X = self.cache["X"]
        Q = self.cache["Q"]
        K = self.cache["K"]
        V = self.cache["V"]

        weights = self.cache["weights"]
        attention_outputs = self.cache["outputs"]
        combined = self.cache["combined"]

        # =========================================
        # Output projection
        # =========================================

        d_W_O = (
            combined.T @ d_output
        )

        d_b_O = np.sum(
            d_output,
            axis=0
        )

        d_combined = (
            d_output @ self.W_O.T
        )

        # =========================================
        # Split combined gradient
        # back into heads
        # =========================================

        d_heads = d_combined.reshape(
            X.shape[0],
            self.num_heads,
            self.head_dim
        )

        d_heads = np.transpose(
            d_heads,
            (1, 0, 2)
        )

        d_Q = np.zeros_like(Q)
        d_K = np.zeros_like(K)
        d_V = np.zeros_like(V)

        # =========================================
        # Backward through each attention head
        # =========================================

        for head in range(self.num_heads):

            q = Q[head]
            k = K[head]
            v = V[head]

            w = weights[head]

            d_output_head = d_heads[head]

            # output = weights @ V
            d_weights = (
                d_output_head @ v.T
            )

            d_v = (
                w.T @ d_output_head
            )

            # =====================================
            # Softmax backward
            # =====================================

            d_scores = np.zeros_like(
                d_weights
            )

            for i in range(len(w)):

                probability = w[i]

                gradient = d_weights[i]

                jacobian = (
                    np.diag(probability)
                    -
                    np.outer(
                        probability,
                        probability
                    )
                )

                d_scores[i] = (
                    jacobian @ gradient
                )

            # =====================================
            # Scaling
            # =====================================

            d_scores /= np.sqrt(
                self.head_dim
            )

            # =====================================
            # scores = Q @ K.T
            # =====================================

            d_q = (
                d_scores @ k
            )

            d_k = (
                d_scores.T @ q
            )

            d_Q[head] = d_q
            d_K[head] = d_k
            d_V[head] = d_v

        # =========================================
        # Combine head gradients
        # =========================================

        d_Q = np.transpose(
            d_Q,
            (1, 0, 2)
        ).reshape(
            X.shape[0],
            self.embedding_dim
        )

        d_K = np.transpose(
            d_K,
            (1, 0, 2)
        ).reshape(
            X.shape[0],
            self.embedding_dim
        )

        d_V = np.transpose(
            d_V,
            (1, 0, 2)
        ).reshape(
            X.shape[0],
            self.embedding_dim
        )

        # =========================================
        # Q = X @ W_Q
        # K = X @ W_K
        # V = X @ W_V
        # =========================================

        d_W_Q = X.T @ d_Q
        d_W_K = X.T @ d_K
        d_W_V = X.T @ d_V

        d_X_Q = d_Q @ self.W_Q.T
        d_X_K = d_K @ self.W_K.T
        d_X_V = d_V @ self.W_V.T

        d_X = (
            d_X_Q
            + d_X_K
            + d_X_V
        )

        return d_X, {
            "W_Q": d_W_Q,
            "W_K": d_W_K,
            "W_V": d_W_V,
            "W_O": d_W_O,
            "b_O": d_b_O
        }