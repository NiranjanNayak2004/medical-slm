import re
from pathlib import Path


INPUT_FILE = Path("data/medical_corpus.txt")
OUTPUT_FILE = Path("data/clean_medical_corpus.txt")


def normalize_text(text):
    # Convert different whitespace characters to spaces
    text = re.sub(r"\s+", " ", text)

    # Keep letters, numbers, basic punctuation
    text = re.sub(r"[^a-zA-Z0-9.,;:!?()/%\- ]", " ", text)

    # Remove repeated punctuation
    text = re.sub(r"([.,;:!?])\1+", r"\1", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Remove excessive spaces again
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_noise(text):
    # Remove very long sequences of numbers
    text = re.sub(r"\b\d{6,}\b", " ", text)

    # Remove common article metadata patterns
    text = re.sub(r"\bdoi:\s*\S+", " ", text, flags=re.IGNORECASE)

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )

    return text


def clean_corpus(text):
    text = remove_noise(text)
    text = normalize_text(text)

    # Lowercase for vocabulary construction
    text = text.lower()

    return text


def main():

    print("=" * 60)
    print("        MedLens Medical Corpus Cleaner")
    print("=" * 60)

    if not INPUT_FILE.exists():
        print("Input file not found:", INPUT_FILE)
        return

    raw_text = INPUT_FILE.read_text(encoding="utf-8")

    raw_words = raw_text.split()

    print("\nRAW CORPUS")
    print("-" * 60)
    print("Characters :", len(raw_text))
    print("Words      :", len(raw_words))

    cleaned_text = clean_corpus(raw_text)

    cleaned_words = cleaned_text.split()

    OUTPUT_FILE.write_text(
        cleaned_text,
        encoding="utf-8"
    )

    print("\nCLEANED CORPUS")
    print("-" * 60)
    print("Characters :", len(cleaned_text))
    print("Words      :", len(cleaned_words))

    print("\nOUTPUT")
    print("-" * 60)
    print(OUTPUT_FILE)

    print("\nSample:")
    print("-" * 60)
    print(cleaned_text[:1000])

    print("\n" + "=" * 60)
    print("Corpus cleaning completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()