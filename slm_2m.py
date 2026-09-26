import numpy as np

from model_config import (
    VOCAB_SIZE,
    EMBEDDING_DIM,
    CONTEXT_LENGTH
)

from transformer_model import TransformerModel
from positional_encoding_v2 import PositionalEncoding


class MedicalSLM:

    def __init__(self):

        # 10,000 × 160 token embedding
        self.embedding_matrix = (
            np.random.randn(
                VOCAB_SIZE,
                EMBEDDING_DIM
            ) * 0.02
        )

        # Positional encoding
        self.position = PositionalEncoding()

        # 2-layer Transformer
        self.transformer = TransformerModel()

        # Output bias for 10,000 vocabulary tokens
        self.output_bias = np.zeros(VOCAB_SIZE)

    def softmax(self, x):

        x = x - np.max(x)

        exp_x = np.exp(x)

        return exp_x / np.sum(exp_x)

    def forward(self, input_ids):

        # ---------------------------------------
        # 1. Token IDs → Embeddings
        # ---------------------------------------

        X = self.embedding_matrix[input_ids]

        # (sequence_length, 160)

        # ---------------------------------------
        # 2. Add positional information
        # ---------------------------------------

        X = self.position.apply(X)

        # (sequence_length, 160)

        # ---------------------------------------
        # 3. Transformer
        # ---------------------------------------

        hidden = self.transformer.forward(X)

        # (sequence_length, 160)

        # ---------------------------------------
        # 4. Take final token
        # ---------------------------------------

        last_hidden = hidden[-1]

        # (160,)

        # ---------------------------------------
        # 5. Weight tying
        # ---------------------------------------

        logits = (
            last_hidden @ self.embedding_matrix.T
            + self.output_bias
        )

        # (10,000,)

        # ---------------------------------------
        # 6. Softmax
        # ---------------------------------------

        probabilities = self.softmax(logits)

        return logits, probabilities


if __name__ == "__main__":

    print("=" * 65)
    print("          MedLens 2M Parameter SLM")
    print("          Complete Forward Pass")
    print("=" * 65)

    # Simulate 128 input tokens
    input_ids = np.random.randint(
        0,
        VOCAB_SIZE,
        size=CONTEXT_LENGTH
    )

    model = MedicalSLM()

    logits, probabilities = model.forward(
        input_ids
    )

    print()
    print("Input IDs          :", input_ids.shape)
    print("Embedding matrix   :", model.embedding_matrix.shape)
    print("Logits             :", logits.shape)
    print("Probabilities      :", probabilities.shape)

    print()
    print(
        "Probability sum    :",
        np.sum(probabilities)
    )

    print()
    print("Top 5 predicted token IDs:")

    top_ids = np.argsort(
        probabilities
    )[-5:][::-1]

    for token_id in top_ids:
        print(
            f"Token {token_id:5d}"
            f" → {probabilities[token_id]:.6f}"
        )

    print()
    print("Complete forward pass working correctly.")