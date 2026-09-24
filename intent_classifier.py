import numpy as np

from tokenizer import tokenize
from slm_model import MedicalSLM


# ==========================================
# Load intent labels
# ==========================================

INTENTS = list(
    np.load(
        "intent_labels.npy",
        allow_pickle=True
    )
)


# ==========================================
# Load trained classifier
# ==========================================

intent_weights = np.load(
    "intent_weights.npy"
)

intent_bias = np.load(
    "intent_bias.npy"
)


# ==========================================
# Load vocabulary
# ==========================================

vocabulary = np.load(
    "transformer_vocabulary.npy",
    allow_pickle=True
).item()


embedding_matrix = np.load(
    "transformer_embedding_matrix.npy"
)


# ==========================================
# Create SLM
# ==========================================

model = MedicalSLM(
    vocabulary_size=len(vocabulary),
    embedding_size=embedding_matrix.shape[1]
)

model.embedding_matrix = embedding_matrix


# ==========================================
# Load Transformer
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
# Get Transformer representation
# ==========================================

def get_representation(text):

    tokens = tokenize(text)

    token_ids = []

    for token in tokens:

        if token in vocabulary:

            token_ids.append(
                vocabulary[token]
            )


    # No known words
    if not token_ids:

        return np.zeros(
            embedding_matrix.shape[1]
        )


    # Use last 3 tokens
    token_ids = token_ids[-3:]


    # Token → embedding
    embeddings = (
        model.embedding_matrix[
            token_ids
        ]
    )


    # Embeddings → Transformer
    transformer_output = (
        model.transformer.forward(
            embeddings
        )
    )


    # Final token representation
    representation = (
        transformer_output[-1]
    )


    # Normalize
    norm = np.linalg.norm(
        representation
    )

    if norm > 0:

        representation = (
            representation / norm
        )


    return representation


# ==========================================
# Predict intent
# ==========================================

def predict_intent(text):

    representation = (
        get_representation(text)
    )


    # Representation → classifier
    logits = (
        representation
        @ intent_weights
        + intent_bias
    )


    # Softmax
    logits = (
        logits - np.max(logits)
    )

    probabilities = (
        np.exp(logits)
        / np.sum(np.exp(logits))
    )


    predicted_id = np.argmax(
        probabilities
    )


    return {
        "intent": INTENTS[predicted_id],
        "confidence": float(
            probabilities[predicted_id]
        )
    }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    test_cases = [

        "I have fever",

        "I am feeling dizzy",

        "I took paracetamol",

        "What is diabetes",

        "I had fever yesterday",

        "I had fever with dizziness and vomiting",

        "My father had fever yesterday",

        "I took paracetamol for fever yesterday"
    ]


    for text in test_cases:

        result = predict_intent(text)

        print("\nInput:", text)

        print(
            "Intent:",
            result["intent"]
        )

        print(
            "Confidence:",
            round(
                result["confidence"],
                4
            )
        )