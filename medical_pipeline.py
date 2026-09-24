from tokenizer import tokenize
from spell_corrector import correct_word
from data_loader import medical_data


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


def extract_facts(tokens):

    facts = []

    for token in tokens:

        if token in medical_data["symptoms"]:

            fact = medical_data["symptoms"][token]

            facts.append({
                "value": token,
                "type": fact["type"],
                "description": fact["description"]
            })

    return facts


def analyze_medical_text(text):

    # -------------------------
    # Step 1: spelling correction
    # -------------------------

    corrected_tokens = correct_text(
        text
    )


    # -------------------------
    # Step 2: fact extraction
    # -------------------------

    facts = extract_facts(
        corrected_tokens
    )


    # -------------------------
    # Step 3: corrected text
    # -------------------------

    corrected_text = " ".join(
        corrected_tokens
    )


    return {
        "original_text": text,
        "corrected_text": corrected_text,
        "facts": facts
    }


# Test

if __name__ == "__main__":

    text = "I have fevr and headche"

    result = analyze_medical_text(
        text
    )

    print("Original:")
    print(result["original_text"])

    print("\nCorrected:")
    print(result["corrected_text"])

    print("\nFacts:")

    for fact in result["facts"]:

        print(fact)