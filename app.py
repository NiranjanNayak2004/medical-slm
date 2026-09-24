from medical_pipeline import analyze_medical_text
from generate import generate_text


def run_medical_slm(text):

    # ======================================
    # 1. Medical analysis
    # ======================================

    analysis = analyze_medical_text(
        text
    )


    # ======================================
    # 2. Generate text using corrected text
    # ======================================

    generated_text = generate_text(
        analysis["corrected_text"],
        number_of_words=5
    )


    # ======================================
    # 3. Return complete result
    # ======================================

    return {
        "original_text": analysis[
            "original_text"
        ],

        "corrected_text": analysis[
            "corrected_text"
        ],

        "facts": analysis[
            "facts"
        ],

        "generated_text": generated_text
    }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    text = "I have fevr and headche"

    result = run_medical_slm(
        text
    )


    print("\n==============================")
    print("MEDICAL SLM")
    print("==============================")

    print("\nOriginal:")
    print(result["original_text"])


    print("\nCorrected:")
    print(result["corrected_text"])


    print("\nMedical Facts:")

    for fact in result["facts"]:

        print(
            f"- {fact['value']} "
            f"→ {fact['type']}"
        )

        print(
            f"  {fact['description']}"
        )


    print("\nGenerated:")
    print(result["generated_text"])