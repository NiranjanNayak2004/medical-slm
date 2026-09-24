import json

from medical_engine import analyze_medical_text


def normalize(value):
    if value is None:
        return None

    return str(value).strip().lower()


def evaluate_case(case):
    text = case["text"]

    result = analyze_medical_text(text)

    # -----------------------------
    # Subject
    # -----------------------------

    detected_subject = normalize(
        result.get("subject")
    )

    expected_subject = normalize(
        case.get("expected_subject")
    )

    subject_correct = (
        expected_subject is None
        or detected_subject == expected_subject
        or (
            expected_subject == "he"
            and detected_subject == "he"
        )
    )

    # -----------------------------
    # Medical entities
    # -----------------------------

    detected_entities = {
        normalize(entity["text"])
        for entity in result.get("medical_entities", [])
    }

    expected_entities = {
        normalize(entity)
        for entity in case.get("expected_entities", [])
    }

    entity_correct = (
        expected_entities == detected_entities
    )

    # -----------------------------
    # Time
    # -----------------------------

    detected_time = normalize(
        result.get("time")
    )

    expected_time = normalize(
        case.get("expected_time")
    )

    time_correct = (
        expected_time is None
        or detected_time == expected_time
    )

    return {
        "text": text,
        "subject": {
            "expected": expected_subject,
            "detected": detected_subject,
            "correct": subject_correct
        },
        "entities": {
            "expected": sorted(expected_entities),
            "detected": sorted(detected_entities),
            "correct": entity_correct
        },
        "time": {
            "expected": expected_time,
            "detected": detected_time,
            "correct": time_correct
        }
    }


def main():

    with open(
        "evaluation_cases.json",
        "r",
        encoding="utf-8"
    ) as file:

        cases = json.load(file)


    results = []

    subject_pass = 0
    entity_pass = 0
    time_pass = 0

    for case in cases:

        result = evaluate_case(case)

        results.append(result)

        if result["subject"]["correct"]:
            subject_pass += 1

        if result["entities"]["correct"]:
            entity_pass += 1

        if result["time"]["correct"]:
            time_pass += 1


    total = len(cases)

    subject_accuracy = (
        subject_pass / total * 100
    )

    entity_accuracy = (
        entity_pass / total * 100
    )

    # Only calculate time accuracy
    # for cases where time is expected.

    time_cases = [
        r for r in results
        if r["time"]["expected"] is not None
    ]

    if time_cases:

        time_correct = sum(
            r["time"]["correct"]
            for r in time_cases
        )

        time_accuracy = (
            time_correct /
            len(time_cases) *
            100
        )

    else:
        time_accuracy = 0


    print()
    print("=" * 60)
    print("MEDICAL SLM PIPELINE EVALUATION")
    print("=" * 60)

    print(f"\nTotal test cases : {total}")

    print(
        f"Subject accuracy : "
        f"{subject_accuracy:.2f}%"
    )

    print(
        f"Entity accuracy  : "
        f"{entity_accuracy:.2f}%"
    )

    print(
        f"Time accuracy    : "
        f"{time_accuracy:.2f}%"
    )


    print("\n" + "-" * 60)
    print("CASE RESULTS")
    print("-" * 60)


    for index, result in enumerate(results, 1):

        subject = "PASS" if result["subject"]["correct"] else "FAIL"
        entity = "PASS" if result["entities"]["correct"] else "FAIL"
        time = "PASS" if result["time"]["correct"] else "FAIL"

        print(f"\n[{index}] {result['text']}")

        print(
            f"  Subject : {subject}"
        )

        print(
            f"  Entities: {entity}"
        )

        print(
            f"  Time    : {time}"
        )

        if not result["entities"]["correct"]:

            print(
                f"  Expected: "
                f"{result['entities']['expected']}"
            )

            print(
                f"  Detected: "
                f"{result['entities']['detected']}"
            )


    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()