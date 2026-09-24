from medical_event_engine_v2 import build_medical_event


test_cases = [
    {
        "name": "Father context",
        "text": (
            "My father had fever yesterday. "
            "He also had dizziness. "
            "He took paracetamol."
        )
    },
    {
        "name": "Mother context",
        "text": (
            "My mother had nausea today. "
            "She also had vomiting. "
            "She took paracetamol."
        )
    },
    {
        "name": "Self context",
        "text": (
            "I had fever yesterday. "
            "I had headache this morning. "
            "I took paracetamol."
        )
    }
]


def main():

    print()
    print("=" * 70)
    print("MEDICAL MULTI-SENTENCE CONTEXT TEST")
    print("=" * 70)

    for index, case in enumerate(test_cases, 1):

        print(f"\n[{index}] {case['name']}")
        print(f"Input: {case['text']}")

        try:

            result = build_medical_event(case["text"])

            print("\nResult:")

            print(result)

        except Exception as error:

            print("\nERROR:")
            print(error)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()