from structured_extractor import (
    extract_structured_data
)

from medical_retriever import (
    retrieve_medical_information
)


text = "I had fevr yesterday with dizziness and vomiting"


# ==========================================
# Extract
# ==========================================

data = extract_structured_data(
    text
)


# ==========================================
# Retrieve
# ==========================================

knowledge = retrieve_medical_information(
    data["medical_entities"]
)


print("\n==============================")

print("Original:")

print(
    data["original_text"]
)


print("\nCorrected:")

print(
    data["corrected_text"]
)


print("\nEntities:")

for entity in data[
    "medical_entities"
]:

    print(entity)


print("\nTime:")

print(
    data.get(
        "time",
        "Not detected"
    )
)


print("\nRetrieved medical information:")

for item in knowledge:

    print("\nEntity:")

    print(
        item["entity"]
    )

    print("\nType:")

    print(
        item["type"]
    )

    print("\nDescription:")

    print(
        item["description"]
    )