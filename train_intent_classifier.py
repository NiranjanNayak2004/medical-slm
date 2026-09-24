import json
import numpy as np

from tokenizer import tokenize
from slm_model import MedicalSLM


# ==========================================
# Intent labels
# ==========================================

INTENTS = [
    "symptom_report",
    "medication",
    "medical_information",
    "symptom_history",
    "medical_event"
]

intent_to_id = {
    intent: i
    for i, intent in enumerate(INTENTS)
}


# ==========================================
# Load dataset
# ==========================================

with open("data/intent_data.json", "r") as file:
    data = json.load(file)


# ==========================================
# Load vocabulary
# ==========================================

vocabulary = np.load(
    "transformer_vocabulary.npy",
    allow_pickle=True
).item()


# ==========================================
# Load embedding matrix
# ==========================================

embedding_matrix = np.load(
    "transformer_embedding_matrix.npy"
)


# ==========================================
# Create model
# ==========================================

model = MedicalSLM(
    len(vocabulary),
    embedding_matrix.shape[1]
)

model.embedding_matrix = embedding_matrix


# ==========================================
# Load Transformer parameters
# ==========================================

parameters = np.load(
    "transformer_parameters.npz"
)

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
# Transformer representation
# ==========================================

def get_representation(text):

    tokens = tokenize(text)

    token_ids = []

    for token in tokens:

        if token in vocabulary:
            token_ids.append(
                vocabulary[token]
            )

    if not token_ids:

        return np.zeros(
            embedding_matrix.shape[1]
        )

    # Same context size used by SLM
    token_ids = token_ids[-3:]

    embeddings = (
        model.embedding_matrix[token_ids]
    )

    transformer_output = (
        model.transformer.forward(
            embeddings
        )
    )

    representation = transformer_output[-1]

    # Normalize representation
    norm = np.linalg.norm(
        representation
    )

    if norm > 0:
        representation = (
            representation / norm
        )

    return representation


# ==========================================
# Prepare training data
# ==========================================

X = []
y = []

for item in data:

    representation = get_representation(
        item["text"]
    )

    X.append(representation)

    y.append(
        intent_to_id[
            item["intent"]
        ]
    )


X = np.array(X)
y = np.array(y)


print("Training examples:", len(X))
print("Representation shape:", X.shape)


# ==========================================
# Classifier
# ==========================================

embedding_size = X.shape[1]

num_classes = len(INTENTS)

np.random.seed(42)

W = (
    np.random.randn(
        embedding_size,
        num_classes
    ) * 0.01
)

b = np.zeros(num_classes)


# ==========================================
# Softmax
# ==========================================

def softmax(x):

    x = x - np.max(x)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)


# ==========================================
# Training
# ==========================================

learning_rate = 0.01

epochs = 1000


for epoch in range(epochs):

    total_loss = 0

    correct = 0


    for i in range(len(X)):

        representation = X[i]

        target = y[i]


        # ------------------------------
        # Forward
        # ------------------------------

        logits = (
            representation @ W + b
        )

        probabilities = softmax(
            logits
        )


        # ------------------------------
        # Prediction
        # ------------------------------

        prediction = np.argmax(
            probabilities
        )

        if prediction == target:
            correct += 1


        # ------------------------------
        # Loss
        # ------------------------------

        loss = -np.log(
            probabilities[target] + 1e-9
        )

        total_loss += loss


        # ------------------------------
        # Gradient
        # ------------------------------

        gradient = probabilities.copy()

        gradient[target] -= 1


        # ------------------------------
        # Update
        # ------------------------------

        W -= learning_rate * np.outer(
            representation,
            gradient
        )

        b -= (
            learning_rate * gradient
        )


    # ==================================
    # Progress
    # ==================================

    if epoch % 100 == 0:

        average_loss = (
            total_loss / len(X)
        )

        accuracy = (
            correct / len(X)
        ) * 100

        print(
            f"Epoch {epoch} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )


# ==========================================
# Save
# ==========================================

np.save(
    "intent_weights.npy",
    W
)

np.save(
    "intent_bias.npy",
    b
)

np.save(
    "intent_labels.npy",
    np.array(INTENTS)
)


print("\nIntent classifier trained.")

print("Saved:")
print("intent_weights.npy")
print("intent_bias.npy")
print("intent_labels.npy")