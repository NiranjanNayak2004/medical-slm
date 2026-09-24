import json
from medical_event_graph import extract_subject


def normalize(value):
    if value is None:
        return None
    return str(value).strip().lower()


def main():

    with open("evaluation_cases.json", "r", encoding="utf-8") as file:
        cases = json.load(file)

    total = len(cases)
    correct = 0

    print()
    print("=" * 60)
    print("MEDICAL SUBJECT EVALUATION")
    print("=" * 60)

    for index, case in enumerate(cases, 1):

        text = case["text"]

        expected = normalize(
            case.get("expected_subject")
        )

        detected = normalize(
            extract_subject(text)
        )

        passed = detected == expected

        if passed:
            correct += 1

        print(f"\n[{index}] {text}")
        print(f"  Expected : {expected}")
        print(f"  Detected : {detected}")
        print(f"  Result   : {'PASS' if passed else 'FAIL'}")

    accuracy = (correct / total) * 100

    print("\n" + "=" * 60)
    print(f"Subject accuracy : {accuracy:.2f}%")
    print(f"Passed           : {correct}/{total}")
    print("=" * 60)


if __name__ == "__main__":
    main()