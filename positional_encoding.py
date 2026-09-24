import numpy as np


def positional_encoding(sequence_length, embedding_size):

    position = np.arange(sequence_length)[:, np.newaxis]

    dimension = np.arange(embedding_size)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (dimension // 2)) / embedding_size
    )

    angles = position * angle_rates

    encoding = np.zeros(
        (sequence_length, embedding_size)
    )

    encoding[:, 0::2] = np.sin(
        angles[:, 0::2]
    )

    encoding[:, 1::2] = np.cos(
        angles[:, 1::2]
    )

    return encoding


# Test

sequence_length = 3
embedding_size = 4

encoding = positional_encoding(
    sequence_length,
    embedding_size
)

if __name__ == "__main__":

    print("Positional encoding:")
    print(positional_encoding(3, 4))