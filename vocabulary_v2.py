import re
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]


class MedicalVocabulary:

    def __init__(self, vocab_size=10_000):
        self.vocab_size = vocab_size

        self.token_to_id = {}
        self.id_to_token = {}

    def tokenize(self, text):
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def build_from_file(self, file_path):

        text = Path(file_path).read_text(
            encoding="utf-8"
        )

        tokens = self.tokenize(text)

        counter = Counter(tokens)

        self.token_to_id = {}
        self.id_to_token = {}

        # Add special tokens first
        for token in SPECIAL_TOKENS:
            token_id = len(self.token_to_id)

            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

        # Remaining vocabulary
        available_tokens = self.vocab_size - len(SPECIAL_TOKENS)

        most_common = counter.most_common(
            available_tokens
        )

        for token, frequency in most_common:

            if token in self.token_to_id:
                continue

            token_id = len(self.token_to_id)

            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

        return counter

    def encode(self, text):

        tokens = self.tokenize(text)

        return [
            self.token_to_id.get(
                token,
                self.token_to_id["<UNK>"]
            )
            for token in tokens
        ]

    def decode(self, token_ids):

        return [
            self.id_to_token.get(
                token_id,
                "<UNK>"
            )
            for token_id in token_ids
        ]

    def save(self, file_path):

        tokens = [
            self.id_to_token[i]
            for i in range(len(self.id_to_token))
        ]

        Path(file_path).write_text(
            "\n".join(tokens),
            encoding="utf-8"
        )


if __name__ == "__main__":

    print("=" * 60)
    print("       MedLens 10K Medical Vocabulary")
    print("=" * 60)

    vocabulary = MedicalVocabulary(
        vocab_size=10_000
    )

    counter = vocabulary.build_from_file(
        "data/clean_medical_corpus.txt"
    )

    print("\nCorpus statistics")
    print("-" * 60)

    print("Unique tokens :", len(counter))
    print("Total tokens  :", sum(counter.values()))

    print("\nVocabulary")
    print("-" * 60)

    print(
        "Vocabulary size:",
        len(vocabulary.token_to_id)
    )

    print(
        "Special tokens:",
        SPECIAL_TOKENS
    )

    print("\nMost frequent medical tokens")
    print("-" * 60)

    for token, frequency in counter.most_common(30):

        print(
            f"{token:<25} {frequency:,}"
        )

    vocabulary.save(
        "data/medical_vocabulary.txt"
    )

    print("\nSaved:")
    print("data/medical_vocabulary.txt")

    print("\nTest encoding")
    print("-" * 60)

    test_text = (
        "The patient has fever and "
        "headache"
    )

    encoded = vocabulary.encode(test_text)

    decoded = vocabulary.decode(encoded)

    print("Text    :", test_text)
    print("Encoded :", encoded)
    print("Decoded :", decoded)

    print("\n" + "=" * 60)
    print("10K vocabulary construction completed.")
    print("=" * 60)