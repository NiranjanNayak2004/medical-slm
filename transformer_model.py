import numpy as np

from model_config import (
    EMBEDDING_DIM,
    NUM_LAYERS,
)
from transformer_block_v2 import TransformerBlock


class TransformerModel:

    def __init__(self):

        # Create 2 Transformer blocks
        self.blocks = [
            TransformerBlock()
            for _ in range(NUM_LAYERS)
        ]

        # Final LayerNorm
        self.gamma = np.ones(EMBEDDING_DIM)
        self.beta = np.zeros(EMBEDDING_DIM)

    def layer_norm(self, x):

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

        normalized = (
            x - mean
        ) / np.sqrt(
            variance + 1e-5
        )

        return self.gamma * normalized + self.beta

    def forward(self, X):

        x = X

        # Pass through Transformer blocks
        for i, block in enumerate(self.blocks):

            x = block.forward(x)

            print(
                f"Transformer Block {i + 1} output: {x.shape}"
            )

        # Final normalization
        x = self.layer_norm(x)

        return x


if __name__ == "__main__":

    print("=" * 60)
    print("        MedLens 2-Layer Transformer Test")
    print("=" * 60)

    sequence_length = 128

    X = np.random.randn(
        sequence_length,
        EMBEDDING_DIM
    )

    model = TransformerModel()

    output = model.forward(X)

    print()
    print("Input shape  :", X.shape)
    print("Output shape :", output.shape)

    print()
    print("Number of Transformer layers:", NUM_LAYERS)
    print("Embedding dimension:", EMBEDDING_DIM)

    print()
    print("2-layer Transformer working correctly.")