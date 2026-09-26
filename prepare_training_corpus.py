import numpy as np
from pathlib import Path

from vocabulary_v2 import MedicalVocabulary
from model_config import CONTEXT_LENGTH


CORPUS_FILE = Path("data/clean_medical_corpus.txt")

X_FILE = Path("data/train_inputs.npy")
Y_FILE = Path("data/train_targets.npy")


def main():

    print("=" * 60)
    print("     MedLens Training Corpus Preparation")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load vocabulary
    # ---------------------------------------------------------

    vocabulary = MedicalVocabulary(vocab_size=10_000)

    vocabulary.build_from_file(
        CORPUS_FILE
    )

    print("\nVocabulary loaded")
    print("Vocabulary size:", len(vocabulary.token_to_id))

    # ---------------------------------------------------------
    # 2. Load medical corpus
    # ---------------------------------------------------------

    text = CORPUS_FILE.read_text(
        encoding="utf-8"
    )

    print("\nCorpus loaded")
    print("Characters:", len(text))
    print("Words:", len(text.split()))

    # ---------------------------------------------------------
    # 3. Convert words to token IDs
    # ---------------------------------------------------------

    token_ids = vocabulary.encode(text)

    token_ids = np.array(
        token_ids,
        dtype=np.int32
    )

    print("\nTokenization")
    print("Total token IDs:", len(token_ids))

    # ---------------------------------------------------------
    # 4. Calculate UNK rate
    # ---------------------------------------------------------

    unk_id = vocabulary.token_to_id["<UNK>"]

    unk_count = np.sum(
        token_ids == unk_id
    )

    unk_rate = (
        unk_count / len(token_ids)
    ) * 100

    print("UNK tokens:", int(unk_count))
    print(f"UNK rate: {unk_rate:.2f}%")

    # ---------------------------------------------------------
    # 5. Create context windows
    # ---------------------------------------------------------

    context_length = CONTEXT_LENGTH

    X = []
    Y = []

    for i in range(
        len(token_ids) - context_length
    ):

        input_sequence = token_ids[
            i:i + context_length
        ]

        target_token = token_ids[
            i + context_length
        ]

        X.append(input_sequence)
        Y.append(target_token)

    X = np.array(
        X,
        dtype=np.int32
    )

    Y = np.array(
        Y,
        dtype=np.int32
    )

    # ---------------------------------------------------------
    # 6. Save training data
    # ---------------------------------------------------------

    np.save(
        X_FILE,
        X
    )

    np.save(
        Y_FILE,
        Y
    )

    # ---------------------------------------------------------
    # 7. Display information
    # ---------------------------------------------------------

    print("\nTraining dataset")
    print("-" * 60)

    print("Context length :", context_length)
    print("Input shape    :", X.shape)
    print("Target shape   :", Y.shape)

    print("\nExample")
    print("-" * 60)

    print("Input token IDs:")
    print(X[0][:20])

    print("\nTarget token ID:")
    print(Y[0])

    print("\nSaved files")
    print("-" * 60)

    print(X_FILE)
    print(Y_FILE)

    print("\n" + "=" * 60)
    print("Training corpus preparation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()