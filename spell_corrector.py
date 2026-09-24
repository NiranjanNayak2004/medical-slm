import json


# ==========================================
# Load medical vocabulary
# ==========================================

with open(
    "data/medical_data.json",
    "r"
) as file:

    medical_data = json.load(file)


medical_terms = list(
    medical_data["symptoms"].keys()
)


# ==========================================
# Edit distance
# ==========================================

def edit_distance(word1, word2):

    rows = len(word1) + 1
    columns = len(word2) + 1

    matrix = [
        [0] * columns
        for _ in range(rows)
    ]

    for i in range(rows):
        matrix[i][0] = i

    for j in range(columns):
        matrix[0][j] = j

    for i in range(1, rows):

        for j in range(1, columns):

            if word1[i - 1] == word2[j - 1]:
                cost = 0
            else:
                cost = 1

            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost
            )

    return matrix[-1][-1]


# ==========================================
# Medical spelling correction
# ==========================================

def correct_word(
    word,
    max_distance=2
):

    # Already a known medical word
    if word in medical_terms:
        return word


    # Very short words should not be
    # automatically corrected
    if len(word) < 4:
        return word


    best_word = word
    best_distance = float("inf")


    for medical_term in medical_terms:

        distance = edit_distance(
            word,
            medical_term
        )

        if distance < best_distance:

            best_distance = distance
            best_word = medical_term


    if best_distance <= max_distance:

        return best_word


    return word