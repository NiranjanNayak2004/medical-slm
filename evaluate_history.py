from medical_history import MedicalHistory


def main():

    history = MedicalHistory()

    conversations = [
        "My father had fever yesterday.",
        "He also had dizziness.",
        "He took paracetamol.",
        "Today he has headache."
    ]

    print()
    print("=" * 70)
    print("MEDICAL HISTORY CONTEXT TEST")
    print("=" * 70)

    for text in conversations:

        print("\nINPUT:")
        print(text)

        try:
            result = history.process(text)

            print("\nRESULT:")
            print(result)

        except Exception as error:
            print("\nERROR:")
            print(error)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()