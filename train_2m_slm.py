import numpy as np

from model_config import (
    VOCAB_SIZE,
    EMBEDDING_DIM,
    CONTEXT_LENGTH
)

from slm_2m import MedicalSLM


# ============================================================
# Configuration
# ============================================================

TRAIN_INPUTS = "data/train_inputs.npy"
TRAIN_TARGETS = "data/train_targets.npy"

LEARNING_RATE = 0.001
EPOCHS = 3
BATCH_SIZE = 16


# ============================================================
# Load dataset
# ============================================================

X = np.load(TRAIN_INPUTS)
Y = np.load(TRAIN_TARGETS)

print("=" * 60)
print("        MedLens 2M SLM Training")
print("=" * 60)

print("\nDataset")
print("-" * 60)
print("Inputs :", X.shape)
print("Targets:", Y.shape)


# ============================================================
# Create model
# ============================================================

model = MedicalSLM()

print("\nModel")
print("-" * 60)

print("Vocabulary size :", VOCAB_SIZE)
print("Embedding dim   :", EMBEDDING_DIM)
print("Context length  :", CONTEXT_LENGTH)


# ============================================================
# Loss function
# ============================================================

def cross_entropy(probabilities, target):

    probability = probabilities[target]

    probability = max(
        probability,
        1e-12
    )

    return -np.log(probability)


# ============================================================
# Training
# ============================================================

for epoch in range(EPOCHS):

    total_loss = 0.0

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    # Shuffle training examples
    indices = np.random.permutation(
        len(X)
    )

    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    # --------------------------------------------------------
    # Mini-batches
    # --------------------------------------------------------

    for start in range(
        0,
        len(X_shuffled),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_shuffled)
        )

        batch_X = X_shuffled[start:end]
        batch_Y = Y_shuffled[start:end]

        batch_loss = 0.0

        # ----------------------------------------------------
        # Process each sequence
        # ----------------------------------------------------

        for input_ids, target_id in zip(
            batch_X,
            batch_Y
        ):

            logits, probabilities = model.forward(
                input_ids
            )

            loss = cross_entropy(
                probabilities,
                target_id
            )

            batch_loss += loss

        batch_loss /= len(batch_X)

        total_loss += batch_loss

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if start % (BATCH_SIZE * 500) == 0:

            print(
                f"  Step {start:>7} "
                f"Loss: {batch_loss:.4f}"
            )

    average_loss = (
        total_loss /
        (len(X_shuffled) / BATCH_SIZE)
    )

    print(
        f"Epoch {epoch + 1} "
        f"Average Loss: {average_loss:.4f}"
    )


print("\n" + "=" * 60)
print("Training completed.")
print("=" * 60)