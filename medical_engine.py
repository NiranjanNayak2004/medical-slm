from structured_extractor import (
    extract_structured_data
)

from relation_extractor import (
    extract_relationships
)

from medical_retriever import (
    retrieve_medical_information
)


def analyze_medical_text(text):

    # ======================================
    # 1. Extract entities + time
    # ======================================

    extracted = extract_structured_data(
        text
    )


    # ======================================
    # 2. Extract relationships
    # ======================================

    relations = extract_relationships(
        text
    )


    # ======================================
    # 3. Retrieve medical information
    # ======================================

    medical_information = (
        retrieve_medical_information(
            extracted["medical_entities"]
        )
    )


    # ======================================
    # 4. Build unified response
    # ======================================

    result = {

        "original_text":
            extracted["original_text"],

        "corrected_text":
            extracted["corrected_text"],

        "medical_entities":
            extracted["medical_entities"],

        "relationships":
            relations["relationships"],

        "medical_information":
            medical_information

    }


    # Add time only when detected

    if "time" in extracted:

        result["time"] = extracted[
            "time"
        ]


    return result


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    text = (
        "I had fevr yesterday "
        "with dizziness and vomiting"
    )


    result = analyze_medical_text(
        text
    )


    print("\n================================")

    print("MEDICAL UNDERSTANDING ENGINE")

    print("================================")


    print("\nOriginal:")

    print(
        result["original_text"]
    )


    print("\nCorrected:")

    print(
        result["corrected_text"]
    )


    print("\nMedical Entities:")

    for entity in result[
        "medical_entities"
    ]:

        print(entity)


    print("\nRelationships:")

    for relationship in result[
        "relationships"
    ]:

        print(relationship)


    print("\nTime:")

    print(
        result.get(
            "time",
            "Not detected"
        )
    )


    print("\nMedical Information:")

    for information in result[
        "medical_information"
    ]:

        print(information)