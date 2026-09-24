from tokenizer import tokenize
from spell_corrector import correct_word
from data_loader import medical_vocabulary


# ==========================================
# Find medical phrases
# ==========================================

def extract_entities(tokens):

    entities = []

    i = 0

    while i < len(tokens):

        found = False


        # ----------------------------------
        # Try longest phrase first
        # ----------------------------------

        for phrase in sorted(
            medical_vocabulary.keys(),
            key=lambda x: len(x.split()),
            reverse=True
        ):

            phrase_tokens = phrase.split()

            phrase_length = len(
                phrase_tokens
            )


            # Not enough tokens remaining

            if i + phrase_length > len(tokens):

                continue


            current_tokens = tokens[
                i:i + phrase_length
            ]


            # Check phrase

            if current_tokens == phrase_tokens:

                information = medical_vocabulary[
                    phrase
                ]


                entities.append({

                    "text": phrase,

                    "category": information[
                        "category"
                    ],

                    "type": information[
                        "type"
                    ]

                })


                # Skip the tokens that
                # belong to this phrase

                i += phrase_length

                found = True

                break


        if not found:

            i += 1


    return entities


# ==========================================
# Correct spelling
# ==========================================

def correct_text(text):

    tokens = tokenize(text)

    corrected_tokens = []

    for token in tokens:

        corrected_token = correct_word(
            token
        )

        corrected_tokens.append(
            corrected_token
        )

    return corrected_tokens


# ==========================================
# Extract time
# ==========================================

def extract_time(text):

    text = text.lower()

    time_patterns = [

        "yesterday",
        "today",
        "tomorrow",

        "last night",
        "this morning",
        "this afternoon",
        "this evening",

        "last week",
        "last month",
        "last year",

        "now",
        "right now",
        "currently",
        "at present"

    ]


    for pattern in time_patterns:

        if pattern in text:

            return pattern


    return None


# ==========================================
# Main extraction
# ==========================================

def extract_structured_data(text):

    # --------------------------------------
    # 1. Correct spelling
    # --------------------------------------

    corrected_tokens = correct_text(
        text
    )


    corrected_text = " ".join(
        corrected_tokens
    )


    # --------------------------------------
    # 2. Extract medical entities
    # --------------------------------------

    entities = extract_entities(
        corrected_tokens
    )


    # --------------------------------------
    # 3. Extract time
    # --------------------------------------

    time = extract_time(
        corrected_text
    )


    # --------------------------------------
    # 4. Build result
    # --------------------------------------

    result = {

        "original_text": text,

        "corrected_text": corrected_text,

        "medical_entities": entities

    }


    if time:

        result["time"] = time


    return result


# ==========================================
# Tests
# ==========================================

if __name__ == "__main__":

    test_cases = [

        "I have common cold",

        "I have high blood pressure",

        "I have chest pain",

        "I have shortness of breath",

        "I had fevr yesterday",

        "He had a fever with dizziness and vomiting"

    ]


    for text in test_cases:

        result = extract_structured_data(
            text
        )


        print(
            "\n=============================="
        )

        print("\nInput:")

        print(
            result["original_text"]
        )


        print("\nCorrected:")

        print(
            result["corrected_text"]
        )


        print("\nMedical entities:")

        for entity in result[
            "medical_entities"
        ]:

            print(entity)


        if "time" in result:

            print("\nTime:")

            print(
                result["time"]
            )