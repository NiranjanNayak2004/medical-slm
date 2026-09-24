from structured_extractor import extract_structured_data


# ==========================================
# Relationship extraction
# ==========================================

def extract_relationships(text):

    data = extract_structured_data(text)

    entities = data["medical_entities"]

    relationships = []


    # --------------------------------------
    # Time relationship
    # --------------------------------------

    if "time" in data:

        for entity in entities:

            relationships.append({
                "subject": entity["text"],
                "relation": "occurred_at",
                "object": data["time"]
            })


    # --------------------------------------
    # Symptom grouping
    # --------------------------------------

    symptoms = []

    for entity in entities:

        if entity["type"] == "symptom":

            symptoms.append(
                entity["text"]
            )


    # --------------------------------------
    # Multiple symptoms
    # --------------------------------------

    if len(symptoms) > 1:

        primary = symptoms[0]

        for symptom in symptoms[1:]:

            relationships.append({
                "subject": primary,
                "relation": "associated_with",
                "object": symptom
            })


    return {
        "original_text": data["original_text"],
        "corrected_text": data["corrected_text"],
        "entities": entities,
        "relationships": relationships
    }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    test_cases = [

        "I had fevr yesterday",

        "He had a fever with dizziness and vomiting",

        "I have chest pain",

        "I have high blood pressure"
    ]


    for text in test_cases:

        result = extract_relationships(
            text
        )


        print(
            "\n=============================="
        )

        print("\nInput:")

        print(
            result["original_text"]
        )


        print("\nCorrected:")

        print(
            result["corrected_text"]
        )


        print("\nEntities:")

        for entity in result[
            "entities"
        ]:

            print(entity)


        print("\nRelationships:")

        for relationship in result[
            "relationships"
        ]:

            print(relationship)