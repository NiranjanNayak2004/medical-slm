def tokenize(text):
    text = text.lower()

    punctuation = ".,!?;:"

    for character in punctuation:
        text = text.replace(character, "")

    return text.split()