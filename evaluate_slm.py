import numpy as np

from tokenizer import tokenize
from slm_model import MedicalSLM


# ==========================================
# 1. Load training text
# ==========================================

with open(
    "data/medical_text.txt",
    "r"
) as file:

    text = file.read()


tokens = tokenize(text)


# ==========================================
# 2. Load vocabulary
# ==========================================

vocabulary = np.load(
    "transformer_vocabulary.npy",
    allow_pickle=True
).item()


id_to_word = {
    value: key
    for key, value in vocabulary.items()
}


# ==========================================
# 3. Convert tokens to IDs
# ==========================================

token_ids = np.array([
    vocabulary[token]
    for token in tokens
])


# ==========================================
# 4. Create model
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

parameters = np.load(
    "transformer_parameters.npz"
)


model = MedicalSLM(
    vocabulary_size=len(vocabulary),
    embedding_size=embedding_matrix.shape[1]
)


# Load trained parameters

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
# 5. Evaluate
# ==========================================

context_size = 3

total_loss = 0

correct_predictions = 0

total_predictions = 0


for i in range(
    len(token_ids) - context_size
):

    input_ids = token_ids[
        i:i + context_size
    ]

    target_id = token_ids[
        i + context_size
    ]


    # Model prediction

    logits, probabilities = model.forward(
        input_ids
    )


    # Cross-entropy loss

    loss = -np.log(
        probabilities[target_id] + 1e-9
    )

    total_loss += loss


    # Predicted token

    predicted_id = np.argmax(
        probabilities
    )


    if predicted_id == target_id:

        correct_predictions += 1


    total_predictions += 1


# ==========================================
# 6. Calculate metrics
# ==========================================

average_loss = (
    total_loss
    / total_predictions
)


accuracy = (
    correct_predictions
    / total_predictions
)


perplexity = np.exp(
    average_loss
)


# ==========================================
# 7. Display results
# ==========================================

print("\n==============================")

print("MEDICAL SLM EVALUATION")

print("==============================")


print(
    f"\nTotal examples: {total_predictions}"
)

print(
    f"Loss: {average_loss:.4f}"
)

print(
    f"Next-token accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Perplexity: {perplexity:.4f}"
)