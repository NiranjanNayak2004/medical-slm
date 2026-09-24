vocabulary = {
    "i": 0,
    "have": 1,
    "fever": 2,
    "headache": 3,
    "cough": 4,
    "and": 5
}

medical_facts = {
    "fever": "symptom",
    "headache": "symptom",
    "cough": "symptom"
}


def tokenize(text):
    text = text.lower()

    punctuation = ".,!?;:"

    for character in punctuation:
        text = text.replace(character, "")

    return text.split()


def edit_distance(word1, word2):

    rows = len(word1) + 1
    columns = len(word2) + 1

    matrix = []

    for i in range(rows):
        matrix.append([0] * columns)

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


def correct_word(word):

    best_word = word
    best_distance = float("inf")

    for known_word in vocabulary:

        distance = edit_distance(word, known_word)

        if distance < best_distance:
            best_distance = distance
            best_word = known_word

    return best_word


def encode(tokens):

    token_ids = []

    for token in tokens:

        if token in vocabulary:
            token_ids.append(vocabulary[token])
        else:
            token_ids.append(-1)

    return token_ids


def extract_facts(tokens):

    facts = []

    for token in tokens:

        if token in medical_facts:
            facts.append({
                "value": token,
                "type": medical_facts[token]
            })

    return facts


sentence = "I have fevr and headche"

tokens = tokenize(sentence)

corrected_tokens = []

for token in tokens:
    corrected_tokens.append(correct_word(token))

token_ids = encode(corrected_tokens)

facts = extract_facts(corrected_tokens)


print("Original:", sentence)
print("Tokens:", tokens)
print("Corrected:", corrected_tokens)
print("Token IDs:", token_ids)
print("Facts:", facts)