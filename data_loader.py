import json


with open(
    "data/medical_data.json",
    "r"
) as file:

    medical_data = json.load(file)


# ==========================================
# Build one medical vocabulary
# ==========================================

medical_vocabulary = {}

for category, entries in medical_data.items():

    for term, information in entries.items():

        medical_vocabulary[term] = {
            "category": category,
            "type": information["type"],
            "description": information["description"]
        }