import json

from medical_history import analyze_medical_history
from condition_state import detect_state


def build_timeline(text):

    history = analyze_medical_history(text)

    timeline = []


    for event in history["events"]:

        # Skip sentences without medical entities
        if not event["entities"]:
            continue


        timeline_event = {

            "subject": event["subject"],

            "time": event["time"] or "unspecified",

            "events": []
        }


        # ------------------------------------------
        # Detect state from the sentence
        # ------------------------------------------

        state = detect_state(
            event["sentence"]
        )


        # ------------------------------------------
        # Add medical entities
        # ------------------------------------------

        for entity in event["entities"]:

            if entity["type"] == "medication":

                relation = "took"

            elif entity["type"] == "symptom":

                relation = "experienced"

            elif entity["type"] == "condition":

                relation = "has_condition"

            else:

                relation = "mentioned"


            timeline_event["events"].append({

                "relation": relation,

                "entity": entity["text"],

                "type": entity["type"],

                "state": state
            })


        timeline.append(
            timeline_event
        )


    return {

        "subject": (
            timeline[0]["subject"]
            if timeline
            else "unknown"
        ),

        "timeline": timeline
    }


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    text = """
    My father had fever yesterday.
    Today he still has dizziness.
    His fever is gone now.
    He took paracetamol this morning.
    """


    result = build_timeline(text)


    print(
        json.dumps(
            result,
            indent=4
        )
    )