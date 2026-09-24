from condition_state import detect_state


tests = [
    "I have fever",
    "I still have dizziness",
    "My fever is gone now",
    "I am feeling better",
    "My symptoms are getting worse",
    "The cough is continuing"
]


for text in tests:

    print(
        text,
        "→",
        detect_state(text)
    )