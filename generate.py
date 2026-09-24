import numpy as np

from tokenizer import tokenize
from slm_model import MedicalSLM


# ==========================================
# 1. Load trained model
# ==========================================

embedding_matrix = np.load(
    "transformer_embedding_matrix.npy"
)

output_weights = np.load(
    "transformer_output_weights.npy"
)

output_bias = np.load(
    "transformer_output_bias.npy"
)

vocabulary = np.load(
    "transformer_vocabulary.npy",
    allow_pickle=True
).item()

parameters = np.load(
    "transformer_parameters.npz"
)


# ==========================================
# 2. Create model
# ==========================================

model = MedicalSLM(
    vocabulary_size=len(vocabulary),
    embedding_size=embedding_matrix.shape[1]
)


# ==========================================
# 3. Load trained parameters
# ==========================================

model.embedding_matrix = embedding_matrix

model.output_weights = output_weights

model.output_bias = output_bias


model.transformer.W_Q = parameters["W_Q"]

model.transformer.W_K = parameters["W_K"]

model.transformer.W_V = parameters["W_V"]


model.transformer.W1 = parameters["W1"]

model.transformer.b1 = parameters["b1"]


model.transformer.W2 = parameters["W2"]

model.transformer.b2 = parameters["b2"]


model.transformer.gamma1 = parameters["gamma1"]

model.transformer.beta1 = parameters["beta1"]


model.transformer.gamma2 = parameters["gamma2"]

model.transformer.beta2 = parameters["beta2"]


# ==========================================
# 4. Reverse vocabulary
# ==========================================

id_to_word = {
    value: key
    for key, value in vocabulary.items()
}


# ==========================================
# 5. Generate text
# ==========================================

def generate_text(
    text,
    number_of_words=5
):

    # Tokenize input
    words = tokenize(text)


    # Words that the SLM actually knows
    model_words = []

    for word in words:

        if word in vocabulary:

            model_words.append(word)


    # Generate one word at a time

    for _ in range(number_of_words):

        # If there are no known words,
        # the SLM cannot generate anything.

        if not model_words:

            break


        # Convert known words to token IDs

        input_ids = [
            vocabulary[word]
            for word in model_words
        ]


        # Model was trained with context size = 3

        context_ids = input_ids[-3:]


        # Predict next word

        predicted_word, probabilities = (
            model.predict_next_word(
                np.array(context_ids),
                id_to_word
            )
        )


        # Add generated word
        # to the final output

        words.append(
            predicted_word
        )


        # Add generated word
        # to the model context

        model_words.append(
            predicted_word
        )


    return " ".join(words)


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    input_text = "I have a"

    result = generate_text(
        input_text,
        number_of_words=5
    )


    print("Input:")

    print(input_text)


    print("\nGenerated text:")

    print(result)