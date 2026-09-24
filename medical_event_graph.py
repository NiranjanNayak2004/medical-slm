from subject_extractor import extract_subject
from structured_extractor import extract_structured_data
from relation_rules import extract_medical_relations


def build_event_graph(text):

    # ==========================================
    # Extract subject
    # ==========================================

    subject = extract_subject(text)


    # ==========================================
    # Extract medical entities
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


    # ==========================================
    # Create graph nodes
    # ==========================================

    nodes = []

    nodes.append({
        "id": subject,
        "type": "person"
    })


    for entity in entities:

        nodes.append({
            "id": entity["text"],
            "type": entity["type"]
        })


    # ==========================================
    # Create relationships
    # ==========================================

    relationships = []


    # Person → medical entity
    for entity in entities:

        if entity["type"] == "medication":

            relation = "took"

        else:

            relation = "experienced"


        relationships.append({

            "subject": subject,

            "relation": relation,

            "object": entity["text"]
        })


    # ==========================================
    # Medical relationships
    # ==========================================

    medical_relations = (
        extract_medical_relations(text)
    )


    relationships.extend(
        medical_relations
    )


    # ==========================================
    # Time relationship
    # ==========================================

    if time:

        relationships.append({

            "subject": subject,

            "relation": "event_time",

            "object": time
        })


    return {

        "nodes": nodes,

        "relationships": relationships
    }


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    tests = [

        "My father had fevr yesterday with dizziness and vomiting",

        "I took paracetamol for fever yesterday",

        "My mother took paracetamol yesterday"
    ]


    for text in tests:

        print("\n")
        print("=" * 70)

        print(text)

        print("=" * 70)

        result = build_event_graph(text)

   
        print(result)