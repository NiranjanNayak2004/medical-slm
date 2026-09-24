import re


def detect_state(text):

    text = text.lower()

    # ------------------------------------------
    # Resolved / ended
    # ------------------------------------------

    resolved_patterns = [
        "is gone",
        "has gone",
        "fever is gone",
        "symptom is gone",
        "no longer",
        "resolved",
        "has resolved",
        "stopped"
    ]

    for pattern in resolved_patterns:

        if pattern in text:
            return "resolved"


    # ------------------------------------------
    # Improving
    # ------------------------------------------

    improving_patterns = [
        "getting better",
        "feeling better",
        "improving",
        "improved",
        "less severe",
        "better now"
    ]

    for pattern in improving_patterns:

        if pattern in text:
            return "improving"


    # ------------------------------------------
    # Worsening
    # ------------------------------------------

    worsening_patterns = [
        "getting worse",
        "worsening",
        "became worse",
        "more severe",
        "increasing"
    ]

    for pattern in worsening_patterns:

        if pattern in text:
            return "worsening"


    # ------------------------------------------
    # Persistent / still active
    # ------------------------------------------

    persistent_patterns = [
        "still",
        "continues",
        "continuing",
        "ongoing",
        "still has",
        "still having"
    ]

    for pattern in persistent_patterns:

        if pattern in text:
            return "persistent"


    # ------------------------------------------
    # Default
    # ------------------------------------------

    return "active"