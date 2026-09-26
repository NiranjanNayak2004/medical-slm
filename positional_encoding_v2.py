import numpy as np

from model_config import CONTEXT_LENGTH, EMBEDDING_DIM


class PositionalEncoding:

    def __init__(self):
        self.encoding = self.create_encoding()

    def create_encoding(self):

        positions = np.arange(
            CONTEXT_LENGTH
        )[:, np.newaxis]

        dimensions = np.arange(
            EMBEDDING_DIM
        )[np.newaxis, :]

        angle_rates = 1 / np.power(
            10000,
            (2 * (dimensions // 2)) / EMBEDDING_DIM
        )

        angles = positions * angle_rates

        encoding = np.zeros(
            (CONTEXT_LENGTH, EMBEDDING_DIM)
        )

        # Even dimensions → sine
        encoding[:, 0::2] = np.sin(
            angles[:, 0::2]
        )

        # Odd dimensions → cosine
        encoding[:, 1::2] = np.cos(
            angles[:, 1::2]
        )

        return encoding

    def apply(self, embeddings):

        sequence_length = embeddings.shape[0]

        return (
            embeddings
            + self.encoding[:sequence_length]
        )


if __name__ == "__main__":

    print("=" * 60)
    print("        Positional Encoding Test")
    print("=" * 60)

    positional_encoding = PositionalEncoding()

    embeddings = np.random.randn(
        128,
        EMBEDDING_DIM
    )

    output = positional_encoding.apply(
        embeddings
    )

    print()
    print("Embedding shape :", embeddings.shape)
    print("Encoding shape  :", positional_encoding.encoding.shape)
    print("Output shape    :", output.shape)

    print()
    print("Position 0 sample:")
    print(positional_encoding.encoding[0][:10])

    print()
    print("Position 1 sample:")
    print(positional_encoding.encoding[1][:10])

    print()
    print("Positional encoding working correctly.")