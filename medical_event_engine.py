import json

from structured_extractor import extract_structured_data
from relation_extractor import extract_relationships
from medical_retriever import retrieve_medical_information
from intent_classifier import predict_intent


def analyze_medical_event(text):

    # ==========================================
    # 1. Intent
    # ==========================================

    intent_result = predict_intent(text)


    # ==========================================
    # 2. Medical entities + spelling + time
    # ==========================================

    extracted = extract_structured_data(text)


    # ==========================================
    # 3. Relationships
    # ==========================================

    relationships = extract_relationships(text)


    # ==========================================
    # 4. Medical knowledge retrieval
    # ==========================================

    medical_information = (
        retrieve_medical_information(
            extracted["medical_entities"]
        )
    )


    # ==========================================
    # 5. Build final result
    # ==========================================

    result = {

        "original_text":
            extracted["original_text"],

        "corrected_text":
            extracted["corrected_text"],

        "intent":
            intent_result["intent"],

        "intent_confidence":
            intent_result["confidence"],

        "medical_entities":
            extracted["medical_entities"],

        "relationships":
            relationships["relationships"],

        "medical_information":
            medical_information
    }


    # ==========================================
    # 6. Time
    # ==========================================

    if "time" in extracted:

        result["time"] = extracted["time"]


    return result


# ==============================================
# TEST
# ==============================================

if __name__ == "__main__":

    test_cases = [

        "I had fevr yesterday",

        "He had a fever with dizziness and vomiting",

        "I took paracetamol for fever yesterday",

        "My father had fevr yesterday with dizziness and vomiting",

        "I have high blood pressure"
    ]


    for text in test_cases:

        print("\n")
        print("=" * 70)

        print("INPUT:")
        print(text)

        print("=" * 70)

        result = analyze_medical_event(text)

        print(
            json.dumps(
                result,
                indent=4
            )
        )