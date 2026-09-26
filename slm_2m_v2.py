import numpy as np

from model_config import (
    VOCAB_SIZE,
    EMBEDDING_DIM
)

from positional_encoding_v2 import PositionalEncoding
from transformer_model_v2 import TransformerModel


class MedicalSLM:

    def __init__(self):

        # ====================================================
        # Input embedding
        # ====================================================

        self.embedding_matrix = (
            np.random.randn(
                VOCAB_SIZE,
                EMBEDDING_DIM
            ) * 0.02
        )

        # ====================================================
        # Positional encoding
        # ====================================================

        self.position = PositionalEncoding()

        # ====================================================
        # Transformer
        # ====================================================

        self.transformer = (
            TransformerModel()
        )

        # ====================================================
        # Output bias
        # ====================================================

        self.output_bias = np.zeros(
            VOCAB_SIZE
        )

        # Cache
        self.cache = None


    # ========================================================
    # Softmax
    # ========================================================

    def softmax(self, x):

        x = x - np.max(x)

        exp_x = np.exp(x)

        return (
            exp_x /
            np.sum(exp_x)
        )


    # ========================================================
    # Forward
    # ========================================================

    def forward(self, input_ids):

        # ------------------------------------
        # 1. Embedding lookup
        # ------------------------------------

        X = self.embedding_matrix[
            input_ids
        ]

        # ------------------------------------
        # 2. Positional encoding
        # ------------------------------------

        X = self.position.apply(X)

        # ------------------------------------
        # 3. Transformer
        # ------------------------------------

        hidden = (
            self.transformer.forward(X)
        )

        # ------------------------------------
        # 4. Last-token representation
        # ------------------------------------

        last_hidden = hidden[-1]

        # ------------------------------------
        # 5. Weight tying
        # ------------------------------------

        logits = (
            last_hidden
            @ self.embedding_matrix.T
            + self.output_bias
        )

        # ------------------------------------
        # 6. Probability distribution
        # ------------------------------------

        probabilities = self.softmax(
            logits
        )

        # Save values for backward
        self.cache = {
            "input_ids": input_ids,
            "X": X,
            "hidden": hidden,
            "last_hidden": last_hidden,
            "probabilities": probabilities
        }

        return logits, probabilities


    # ========================================================
    # Cross-Entropy Loss
    # ========================================================

    def loss(self, target_id):

        probabilities = (
            self.cache["probabilities"]
        )

        probability = max(
            probabilities[target_id],
            1e-12
        )

        return -np.log(probability)


    # ========================================================
    # Backward
    # ========================================================

    def backward(self, target_id):

        input_ids = self.cache["input_ids"]
        hidden = self.cache["hidden"]
        last_hidden = self.cache["last_hidden"]
        probabilities = self.cache["probabilities"]

        # ====================================================
        # Cross entropy + softmax gradient
        # ====================================================

        d_logits = probabilities.copy()

        d_logits[target_id] -= 1.0

        # ====================================================
        # Output layer
        #
        # logits =
        # last_hidden @ embedding_matrix.T
        # ====================================================

        d_last_hidden = (
            d_logits
            @ self.embedding_matrix
        )

        # ====================================================
        # Gradient from weight tying
        #
        # embedding_matrix receives gradient
        # from the output projection.
        # ====================================================

        d_embedding_output = np.outer(
            d_logits,
            last_hidden
        )

        # ====================================================
        # Gradient through Transformer
        # ====================================================

        d_hidden = np.zeros_like(
            hidden
        )

        d_hidden[-1] = d_last_hidden

        d_X, transformer_gradients = (
            self.transformer.backward(
                d_hidden
            )
        )

        # ====================================================
        # Gradient through embedding lookup
        # ====================================================

        d_embedding_input = np.zeros_like(
            self.embedding_matrix
        )

        for position, token_id in enumerate(
            input_ids
        ):

            d_embedding_input[
                token_id
            ] += d_X[position]

        # ====================================================
        # Combine both embedding gradients
        #
        # This is weight tying.
        # ====================================================

        d_embedding = (
            d_embedding_input
            + d_embedding_output
        )

        # ====================================================
        # Output bias
        # ====================================================

        d_output_bias = d_logits

        gradients = {

            "embedding_matrix":
                d_embedding,

            "output_bias":
                d_output_bias,

            "transformer":
                transformer_gradients
        }

        return gradients