import numpy as np

from slm_2m_v2 import MedicalSLM
from model_config import (
    CONTEXT_LENGTH,
    VOCAB_SIZE,
    EMBEDDING_DIM
)


print("=" * 60)
print("        MedLens 2M SLM — Gradient Test")
print("=" * 60)


# ------------------------------------------------------------
# 1. Create model
# ------------------------------------------------------------

model = MedicalSLM()

print("\nModel created")


# ------------------------------------------------------------
# 2. Create a small fake input
# ------------------------------------------------------------

input_ids = np.random.randint(
    0,
    VOCAB_SIZE,
    size=CONTEXT_LENGTH
)

target_id = np.random.randint(
    0,
    VOCAB_SIZE
)

print("\nInput")
print("-" * 60)
print("Sequence length :", len(input_ids))
print("Target token    :", target_id)


# ------------------------------------------------------------
# 3. Forward pass
# ------------------------------------------------------------

logits, probabilities = model.forward(
    input_ids
)

print("\nForward pass")
print("-" * 60)

print("Logits shape        :", logits.shape)
print("Probabilities shape :", probabilities.shape)
print("Probability sum     :", np.sum(probabilities))


# ------------------------------------------------------------
# 4. Loss
# ------------------------------------------------------------

loss = model.loss(
    target_id
)

print("\nLoss")
print("-" * 60)
print("Cross entropy:", loss)


# ------------------------------------------------------------
# 5. Backward pass
# ------------------------------------------------------------

gradients = model.backward(
    target_id
)

print("\nBackward pass completed")


# ------------------------------------------------------------
# 6. Check gradients
# ------------------------------------------------------------

embedding_gradient = (
    gradients["embedding_matrix"]
)

print("\nEmbedding gradient")
print("-" * 60)

print(
    "Shape:",
    embedding_gradient.shape
)

print(
    "Mean:",
    np.mean(embedding_gradient)
)

print(
    "Max:",
    np.max(np.abs(embedding_gradient))
)


# ------------------------------------------------------------
# 7. Check for NaN / Inf
# ------------------------------------------------------------

has_nan = False
has_inf = False


def check_array(name, array):

    global has_nan
    global has_inf

    if np.any(np.isnan(array)):
        print("NaN detected:", name)
        has_nan = True

    if np.any(np.isinf(array)):
        print("Inf detected:", name)
        has_inf = True


check_array(
    "logits",
    logits
)

check_array(
    "probabilities",
    probabilities
)

check_array(
    "embedding_gradient",
    embedding_gradient
)


# ------------------------------------------------------------
# 8. Final result
# ------------------------------------------------------------

print("\n" + "=" * 60)

if has_nan or has_inf:

    print("GRADIENT TEST FAILED")

else:

    print("GRADIENT TEST PASSED")

print("=" * 60)