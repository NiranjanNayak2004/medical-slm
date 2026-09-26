# ============================================================
# MedLens 2M Parameter SLM
# Model Configuration
# ============================================================

VOCAB_SIZE = 10_000

EMBEDDING_DIM = 160

NUM_LAYERS = 2

NUM_HEADS = 4

HEAD_DIM = EMBEDDING_DIM // NUM_HEADS

FFN_DIM = 320

CONTEXT_LENGTH = 128


# ============================================================
# VALIDATION
# ============================================================

assert EMBEDDING_DIM % NUM_HEADS == 0, (
    "Embedding dimension must be divisible by number of heads."
)


def print_config():

    print("=" * 55)
    print("        MedLens Medical SLM — Configuration")
    print("=" * 55)

    print(f"Vocabulary size      : {VOCAB_SIZE:,}")
    print(f"Embedding dimension  : {EMBEDDING_DIM}")
    print(f"Transformer layers   : {NUM_LAYERS}")
    print(f"Attention heads      : {NUM_HEADS}")
    print(f"Head dimension       : {HEAD_DIM}")
    print(f"FFN dimension        : {FFN_DIM}")
    print(f"Context length       : {CONTEXT_LENGTH}")

    print("=" * 55)


if __name__ == "__main__":
    print_config()