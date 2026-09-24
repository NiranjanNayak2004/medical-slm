import re


class MedicalContext:

    def __init__(self):
        self.current_subject = None


    def update_subject(self, subject):

        if subject and subject != "unknown":
            self.current_subject = subject


    def resolve_subject(self, text, detected_subject):

        text_lower = text.lower()

        # ------------------------------------------
        # Pronouns that refer to previous subject
        # ------------------------------------------

        pronouns = [
            "he",
            "him",
            "his",
            "she",
            "her",
            "they",
            "them",
            "their"
        ]

        # If detected subject is a pronoun,
        # resolve it using previous context.

        if detected_subject in pronouns:

            if self.current_subject:
                return self.current_subject


        # Also check pronouns directly in sentence

        for pronoun in pronouns:

            if re.search(
                r"\b" + pronoun + r"\b",
                text_lower
            ):

                if self.current_subject:
                    return self.current_subject


        # ------------------------------------------
        # Explicit family/person subjects
        # ------------------------------------------

        explicit_subjects = [
            "father",
            "mother",
            "brother",
            "sister"
        ]

        for person in explicit_subjects:

            if re.search(
                r"\bmy\s+" + person + r"\b",
                text_lower
            ):

                self.current_subject = person

                return person

            if re.search(
                r"\b" + person + r"\b",
                text_lower
            ):

                self.current_subject = person

                return person


        # ------------------------------------------
        # First-person reference
        # ------------------------------------------

        if re.search(
            r"\b(i|me|myself)\b",
            text_lower
        ):

            self.current_subject = "I"

            return "I"


        # ------------------------------------------
        # Explicit detected subject
        # ------------------------------------------

        if detected_subject != "unknown":

            self.current_subject = detected_subject

            return detected_subject


        # ------------------------------------------
        # Previous context
        # ------------------------------------------

        return self.current_subject or "unknown"