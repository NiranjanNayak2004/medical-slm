import re


SUBJECTS = {
    "i": "I",
    "me": "I",
    "myself": "I",
    "he": "he",
    "him": "he",
    "his": "he",
    "she": "she",
    "her": "she",
    "they": "they",
    "them": "they",
    "my father": "father",
    "my mother": "mother",
    "my brother": "brother",
    "my sister": "sister"
}


def extract_subject(text):

    text = text.lower()

    # Check longer phrases first
    for phrase in sorted(
        SUBJECTS,
        key=len,
        reverse=True
    ):

        pattern = r"\b" + re.escape(phrase) + r"\b"

        if re.search(pattern, text):

            return SUBJECTS[phrase]

    return "unknown"


if __name__ == "__main__":

    tests = [
        "I had fever",
        "He had fever",
        "She has headache",
        "My father had fever",
        "My mother took medicine",
        "They have dizziness"
    ]

    for text in tests:

        print(
            text,
            "→",
            extract_subject(text)
        )