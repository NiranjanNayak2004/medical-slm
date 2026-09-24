from subject_extractor import extract_subject
from structured_extractor import extract_structured_data
from relation_rules import extract_medical_relations


def build_medical_event(text):

    # ==========================================
    # Extract information
    # ==========================================

    subject = extract_subject(text)

    extracted = extract_structured_data(text)

    entities = extracted.get(
        "medical_entities",
        []
    )

    time = extracted.get(
        "time",
        None
    )

    # ==========================================
    # Create event
    # ==========================================

    event = {
        "subject": subject,
        "time": time,
        "medical_entities": entities,
        "relationships": []
    }

    # ==========================================
    # Subject → entity relationships
    # ==========================================

    for entity in entities:

        if entity["type"] == "medication":
            relation = "took"
        else:
            relation = "experienced"

        event["relationships"].append({

            "subject": subject,
            "relation": relation,
            "object": entity["text"]

        })

    # ==========================================
    # Medical relationships
    # ==========================================

    medical_relations = extract_medical_relations(text)

    # ==========================================
    # Add only valid medical relationships
    # ==========================================

    for relation in medical_relations:

        relation_type = relation.get("relation")

        # --------------------------------------
        # Treatment relationship
        # --------------------------------------
        #
        # Only accept "treated" when the
        # relationship is explicitly present
        # in the original sentence.
        #
        # Example:
        #
        # "I took paracetamol for fever"
        #
        # is valid.
        #
        # But:
        #
        # "I had dizziness. I took paracetamol."
        #
        # should NOT automatically mean
        # paracetamol treated dizziness.
        # --------------------------------------

        if relation_type == "treated":

            relation_text = text.lower()

            medication = relation.get(
                "subject",
                ""
            ).lower()

            condition = relation.get(
                "object",
                ""
            ).lower()

            explicit_treatment = (
                f"{medication} for {condition}"
                in relation_text
            )

            if not explicit_treatment:
                continue

        event["relationships"].append(
            relation
        )

    return event


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    tests = [

        "My father had fevr yesterday with dizziness and vomiting",

        "I took paracetamol for fever yesterday",

        "My mother took paracetamol yesterday",

        "My father had fever yesterday. "
        "He also had dizziness. "
        "He took paracetamol."

    ]

    for text in tests:

        print("\n")
        print("=" * 70)

        print(text)

        print("=" * 70)

        print(
            build_medical_event(text)
        )