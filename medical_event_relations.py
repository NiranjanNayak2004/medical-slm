from subject_extractor import extract_subject
from structured_extractor import extract_structured_data


def build_medical_events(text):

    # ==========================================
    # Extract subject
    # ==========================================

    subject = extract_subject(text)


    # ==========================================
    # Extract medical information
    # ==========================================

    extracted = extract_structured_data(text)

    entities = extracted[
        "medical_entities"
    ]


    # ==========================================
    # Extract time
    # ==========================================

    time = extracted.get(
        "time",
        None
    )


    events = []


    # ==========================================
    # Create events
    # ==========================================

    for entity in entities:

        entity_text = entity["text"]

        entity_type = entity["type"]


        # --------------------------------------
        # Medication
        # --------------------------------------

        if entity_type == "medication":

            relation = "took"

        # --------------------------------------
        # Symptoms / conditions
        # --------------------------------------

        else:

            relation = "experienced"


        event = {

            "subject": subject,

            "relation": relation,

            "entity": entity_text,

            "type": entity_type
        }


        # Add time if available

        if time:

            event["time"] = time


        events.append(event)


    return events


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    test_cases = [

        "I had fever yesterday",

        "My father had fevr yesterday",

        "He had a fever with dizziness and vomiting",

        "My mother took paracetamol yesterday"
    ]


    for text in test_cases:

        print("\nInput:")
        print(text)

        print("\nEvents:")

        for event in build_medical_events(text):

            print(event)