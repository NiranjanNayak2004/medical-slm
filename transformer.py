import numpy as np
from positional_encoding import positional_encoding


class TransformerBlock:

    def __init__(self, embedding_size):

        self.embedding_size = embedding_size

        # Attention weights
        self.W_Q = np.random.randn(
            embedding_size, embedding_size
        ) * 0.1

        self.W_K = np.random.randn(
            embedding_size, embedding_size
        ) * 0.1

        self.W_V = np.random.randn(
            embedding_size, embedding_size
        ) * 0.1

        # Feed-forward network
        self.W1 = np.random.randn(
            embedding_size,
            embedding_size * 2
        ) * 0.1

        self.b1 = np.zeros(
            embedding_size * 2
        )

        self.W2 = np.random.randn(
            embedding_size * 2,
            embedding_size
        ) * 0.1

        self.b2 = np.zeros(
            embedding_size
        )

        # LayerNorm parameters
        self.gamma1 = np.ones(
            embedding_size
        )

        self.beta1 = np.zeros(
            embedding_size
        )

        self.gamma2 = np.ones(
            embedding_size
        )

        self.beta2 = np.zeros(
            embedding_size
        )


    def softmax(self, x):

        exp_x = np.exp(
            x - np.max(
                x,
                axis=-1,
                keepdims=True
            )
        )

        return exp_x / np.sum(
            exp_x,
            axis=-1,
            keepdims=True
        )


    def layer_norm(
        self,
        x,
        gamma,
        beta
    ):

        mean = np.mean(
            x,
            axis=-1,
            keepdims=True
        )

        variance = np.var(
            x,
            axis=-1,
            keepdims=True
        )

        std = np.sqrt(
            variance + 1e-5
        )

        normalized = (
            x - mean
        ) / std

        output = (
            gamma * normalized
            + beta
        )

        return output


    def layer_norm_backward(
        self,
        grad_output,
        x,
        gamma
    ):

        n = x.shape[-1]

        mean = np.mean(
            x,
            axis=-1,
            keepdims=True
        )

        variance = np.var(
            x,
            axis=-1,
            keepdims=True
        )

        std = np.sqrt(
            variance + 1e-5
        )

        normalized = (
            x - mean
        ) / std

        grad_gamma = np.sum(
            grad_output * normalized,
            axis=0
        )

        grad_beta = np.sum(
            grad_output,
            axis=0
        )

        dx = (
            gamma
            / std
            / n
            * (
                n * grad_output
                - np.sum(
                    grad_output,
                    axis=-1,
                    keepdims=True
                )
                - normalized
                * np.sum(
                    grad_output * normalized,
                    axis=-1,
                    keepdims=True
                )
            )
        )

        return dx, grad_gamma, grad_beta


    def forward(
        self,
        X,
        return_cache=False
    ):

        sequence_length = X.shape[0]

        # -------------------------
        # Positional encoding
        # -------------------------

        position = positional_encoding(
            sequence_length,
            self.embedding_size
        )

        H = X + position


        # -------------------------
        # LayerNorm 1
        # -------------------------

        N1 = self.layer_norm(
            H,
            self.gamma1,
            self.beta1
        )


        # -------------------------
        # Q K V
        # -------------------------

        Q = np.dot(
            N1,
            self.W_Q
        )

        K = np.dot(
            N1,
            self.W_K
        )

        V = np.dot(
            N1,
            self.W_V
        )


        # -------------------------
        # Attention scores
        # -------------------------

        scores = np.dot(
            Q,
            K.T
        )

        scores = scores / np.sqrt(
            self.embedding_size
        )


        # -------------------------
        # Causal mask
        # -------------------------

        mask = np.triu(
            np.ones(
                (
                    sequence_length,
                    sequence_length
                )
            ),
            k=1
        )

        scores = np.where(
            mask == 1,
            -1e9,
            scores
        )


        # -------------------------
        # Attention
        # -------------------------

        A = self.softmax(
            scores
        )

        C = np.dot(
            A,
            V
        )


        # -------------------------
        # Residual 1
        # -------------------------

        R1 = H + C


        # -------------------------
        # LayerNorm 2
        # -------------------------

        N2 = self.layer_norm(
            R1,
            self.gamma2,
            self.beta2
        )


        # -------------------------
        # Feed-forward
        # -------------------------

        F1 = np.dot(
            N2,
            self.W1
        ) + self.b1

        R = np.maximum(
            0,
            F1
        )

        F2 = np.dot(
            R,
            self.W2
        ) + self.b2


        # -------------------------
        # Residual 2
        # -------------------------

        Y = R1 + F2


        if return_cache:

            cache = {
                "X": X,
                "H": H,
                "N1": N1,
                "Q": Q,
                "K": K,
                "V": V,
                "A": A,
                "R1": R1,
                "N2": N2,
                "F1": F1,
                "R": R
            }

            return Y, cache

        return Y


    def backward(
        self,
        grad_output,
        cache
    ):

        X = cache["X"]
        H = cache["H"]
        N1 = cache["N1"]
        Q = cache["Q"]
        K = cache["K"]
        V = cache["V"]
        A = cache["A"]
        R1 = cache["R1"]
        N2 = cache["N2"]
        F1 = cache["F1"]
        R = cache["R"]


        # ==================================================
        # Feed-forward backward
        # ==================================================

        # Y = R1 + F2

        grad_R1 = grad_output.copy()

        grad_F2 = grad_output


        # F2 = R @ W2 + b2

        grad_W2 = np.dot(
            R.T,
            grad_F2
        )

        grad_b2 = np.sum(
            grad_F2,
            axis=0
        )

        grad_R = np.dot(
            grad_F2,
            self.W2.T
        )


        # ReLU backward

        grad_F1 = (
            grad_R
            * (F1 > 0)
        )


        # F1 = N2 @ W1 + b1

        grad_W1 = np.dot(
            N2.T,
            grad_F1
        )

        grad_b1 = np.sum(
            grad_F1,
            axis=0
        )

        grad_N2 = np.dot(
            grad_F1,
            self.W1.T
        )


        # LayerNorm 2 backward

        grad_R1_norm, grad_gamma2, grad_beta2 = (
            self.layer_norm_backward(
                grad_N2,
                R1,
                self.gamma2
            )
        )

        grad_R1 += grad_R1_norm


        # ==================================================
        # Attention backward
        # ==================================================

        # R1 = H + C

        grad_H = grad_R1.copy()

        grad_C = grad_R1


        # C = A @ V

        grad_A = np.dot(
            grad_C,
            V.T
        )

        grad_V = np.dot(
            A.T,
            grad_C
        )


        # Softmax backward

        grad_scores = np.zeros_like(
            A
        )

        for i in range(
            A.shape[0]
        ):

            a = A[i]

            jacobian = (
                np.diag(a)
                - np.outer(a, a)
            )

            grad_scores[i] = np.dot(
                jacobian,
                grad_A[i]
            )


        # Remove future-token gradients

        sequence_length = A.shape[0]

        mask = np.triu(
            np.ones(
                (
                    sequence_length,
                    sequence_length
                )
            ),
            k=1
        )

        grad_scores = np.where(
            mask == 1,
            0,
            grad_scores
        )


        # scores = Q @ K.T / sqrt(d)

        scale = np.sqrt(
            self.embedding_size
        )

        grad_Q = np.dot(
            grad_scores,
            K
        ) / scale

        grad_K = np.dot(
            grad_scores.T,
            Q
        ) / scale


        # Q = N1 @ W_Q

        grad_W_Q = np.dot(
            N1.T,
            grad_Q
        )

        grad_W_K = np.dot(
            N1.T,
            grad_K
        )

        grad_W_V = np.dot(
            N1.T,
            grad_V
        )


        grad_N1 = (
            np.dot(
                grad_Q,
                self.W_Q.T
            )
            +
            np.dot(
                grad_K,
                self.W_K.T
            )
            +
            np.dot(
                grad_V,
                self.W_V.T
            )
        )


        # LayerNorm 1 backward

        grad_H_norm, grad_gamma1, grad_beta1 = (
            self.layer_norm_backward(
                grad_N1,
                H,
                self.gamma1
            )
        )

        grad_H += grad_H_norm


        # H = X + positional_encoding

        grad_X = grad_H


        gradients = {

            "W_Q": grad_W_Q,
            "W_K": grad_W_K,
            "W_V": grad_W_V,

            "W1": grad_W1,
            "b1": grad_b1,

            "W2": grad_W2,
            "b2": grad_b2,

            "gamma1": grad_gamma1,
            "beta1": grad_beta1,

            "gamma2": grad_gamma2,
            "beta2": grad_beta2
        }

        return grad_X, gradients


if __name__ == "__main__":

    X = np.array([
        [1.0, 0.0, 1.0, 0.5],
        [0.0, 1.0, 1.0, 0.2],
        [1.0, 1.0, 0.0, 0.8]
    ])

    model = TransformerBlock(
        embedding_size=4
    )

    output = model.forward(X)

    print("Output shape:")
    print(output.shape)

    print("\nTransformer output:")
    print(output)