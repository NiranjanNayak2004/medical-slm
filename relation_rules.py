from structured_extractor import extract_structured_data


def extract_medical_relations(text):

    extracted = extract_structured_data(text)

    entities = extracted["medical_entities"]

    relationships = []

    medications = [
        e["text"]
        for e in entities
        if e["type"] == "medication"
    ]

    symptoms = [
        e["text"]
        for e in entities
        if e["type"] == "symptom"
    ]

    conditions = [
        e["text"]
        for e in entities
        if e["type"] == "condition"
    ]

    # ------------------------------------------
    # Medication → symptom
    # ------------------------------------------

    if medications and symptoms:

        for medication in medications:

            for symptom in symptoms:

                relationships.append({
                    "subject": medication,
                    "relation": "treated",
                    "object": symptom
                })

    # ------------------------------------------
    # Medication → condition
    # ------------------------------------------

    if medications and conditions:

        for medication in medications:

            for condition in conditions:

                relationships.append({
                    "subject": medication,
                    "relation": "used_for",
                    "object": condition
                })

    # ------------------------------------------
    # Symptoms associated with each other
    # ------------------------------------------

    if len(symptoms) > 1:

        primary = symptoms[0]

        for symptom in symptoms[1:]:

            relationships.append({
                "subject": primary,
                "relation": "associated_with",
                "object": symptom
            })

    return relationships


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    tests = [

        "I took paracetamol for fever",

        "He had fever with dizziness and vomiting",

        "I took ibuprofen for headache",

        "I have fever and headache"
    ]

    for text in tests:

        print("\nInput:")
        print(text)

        print("\nRelationships:")

        relationships = (
            extract_medical_relations(text)
        )

        for relationship in relationships:

            print(relationship)