import re
import json

from subject_extractor import extract_subject
from structured_extractor import extract_structured_data
from relation_rules import extract_medical_relations
from context_manager import MedicalContext


def split_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def analyze_medical_history(text):

    context = MedicalContext()

    events = []


    sentences = split_sentences(text)


    for sentence in sentences:

        # --------------------------------------
        # Detect subject
        # --------------------------------------

        detected_subject = extract_subject(
            sentence
        )


        # --------------------------------------
        # Resolve subject using context
        # --------------------------------------

        subject = context.resolve_subject(
            sentence,
            detected_subject
        )


        # --------------------------------------
        # Extract medical information
        # --------------------------------------

        extracted = extract_structured_data(
            sentence
        )


        entities = extracted[
            "medical_entities"
        ]


        time = extracted.get(
            "time",
            None
        )


        # --------------------------------------
        # Create event
        # --------------------------------------

        event = {

            "sentence": sentence,

            "subject": subject,

            "entities": entities,

            "time": time,

            "relationships": []
        }


        # --------------------------------------
        # Subject → entity
        # --------------------------------------

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


        # --------------------------------------
        # Medical relationships
        # --------------------------------------

        medical_relations = (
            extract_medical_relations(
                sentence
            )
        )


        event["relationships"].extend(
            medical_relations
        )


        events.append(event)


    return {

        "original_text": text,

        "events": events
    }


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    text = """
    My father had fever yesterday.
    He also had dizziness.
    He took paracetamol.
    Today he is feeling better.
    """


    result = analyze_medical_history(
        text
    )


    print(
        json.dumps(
            result,
            indent=4
        )
    )