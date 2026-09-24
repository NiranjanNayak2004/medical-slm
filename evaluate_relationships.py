import json
from medical_event_graph import build_event_graph


def normalize(value):
    return str(value).strip().lower()


def relation_key(relation):
    return (
        normalize(relation.get("subject", "")),
        normalize(relation.get("relation", "")),
        normalize(relation.get("object", ""))
    )


def main():

    with open("evaluation_cases.json", "r", encoding="utf-8") as file:
        cases = json.load(file)

    total_cases = len(cases)
    cases_with_relationships = 0

    print()
    print("=" * 60)
    print("MEDICAL RELATIONSHIP EVALUATION")
    print("=" * 60)

    for index, case in enumerate(cases, 1):

        text = case["text"]

        result = build_event_graph(text)

        relationships = result.get("relationships", [])

        print(f"\n[{index}] {text}")

        if relationships:
            cases_with_relationships += 1

            for relation in relationships:
                print(
                    f"  {relation.get('subject')} "
                    f"→ {relation.get('relation')} "
                    f"→ {relation.get('object')}"
                )

        else:
            print("  No relationships detected.")

    print("\n" + "=" * 60)

    relationship_rate = (
        cases_with_relationships / total_cases * 100
    )

    print(
        f"Cases with relationships : "
        f"{cases_with_relationships}/{total_cases}"
    )

    print(
        f"Relationship detection rate : "
        f"{relationship_rate:.2f}%"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()