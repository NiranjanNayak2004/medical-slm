# ============================================================
# MedLens Medical SLM
# 2M Parameter Counter
# ============================================================

from model_config import (
    VOCAB_SIZE,
    EMBEDDING_DIM,
    NUM_LAYERS,
    NUM_HEADS,
    FFN_DIM,
)


# ============================================================
# PARAMETER COUNTING
# ============================================================

def count_parameters():

    # --------------------------------------------------------
    # 1. Token Embedding
    # --------------------------------------------------------

    embedding = VOCAB_SIZE * EMBEDDING_DIM


    # --------------------------------------------------------
    # 2. Attention
    # --------------------------------------------------------

    # Query, Key, Value
    qkv_weights = 3 * EMBEDDING_DIM * EMBEDDING_DIM

    qkv_biases = 3 * EMBEDDING_DIM


    # Output projection of multi-head attention
    attention_output_weights = (
        EMBEDDING_DIM * EMBEDDING_DIM
    )

    attention_output_bias = EMBEDDING_DIM

    attention = (
        qkv_weights
        + qkv_biases
        + attention_output_weights
        + attention_output_bias
    )


    # --------------------------------------------------------
    # 3. Feed Forward Network
    # --------------------------------------------------------

    ffn_weights = (
        EMBEDDING_DIM * FFN_DIM
        +
        FFN_DIM * EMBEDDING_DIM
    )

    ffn_biases = (
        FFN_DIM
        +
        EMBEDDING_DIM
    )

    ffn = (
        ffn_weights
        + ffn_biases
    )


    # --------------------------------------------------------
    # 4. LayerNorm
    # --------------------------------------------------------

    # Two LayerNorm layers per Transformer block
    layer_norm = (
        2
        * EMBEDDING_DIM
        * 2
    )


    # --------------------------------------------------------
    # 5. One Transformer block
    # --------------------------------------------------------

    one_block = (
        attention
        + ffn
        + layer_norm
    )


    # --------------------------------------------------------
    # 6. All Transformer blocks
    # --------------------------------------------------------

    transformer = (
        NUM_LAYERS
        * one_block
    )


    # --------------------------------------------------------
    # 7. Final LayerNorm
    # --------------------------------------------------------

    final_layer_norm = (
        2 * EMBEDDING_DIM
    )


    # --------------------------------------------------------
    # 8. Output bias
    #
    # Output projection shares the embedding matrix
    # (weight tying), so there is NO second 160 x 10000
    # weight matrix.
    # --------------------------------------------------------

    output_bias = VOCAB_SIZE


    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    total = (
        embedding
        + transformer
        + final_layer_norm
        + output_bias
    )


    return {
        "embedding": embedding,
        "attention_per_block": attention,
        "ffn_per_block": ffn,
        "layer_norm_per_block": layer_norm,
        "one_transformer_block": one_block,
        "all_transformer_blocks": transformer,
        "final_layer_norm": final_layer_norm,
        "output_bias": output_bias,
        "total": total,
    }


# ============================================================
# DISPLAY
# ============================================================

def print_parameter_report():

    params = count_parameters()

    print("=" * 65)
    print("             MedLens Medical SLM")
    print("             Parameter Report")
    print("=" * 65)

    print()
    print(f"Vocabulary size       : {VOCAB_SIZE:,}")
    print(f"Embedding dimension   : {EMBEDDING_DIM}")
    print(f"Transformer layers    : {NUM_LAYERS}")
    print(f"Attention heads       : {NUM_HEADS}")
    print(f"FFN dimension         : {FFN_DIM}")

    print()
    print("-" * 65)

    print(
        f"Token embedding       : {params['embedding']:,}"
    )

    print(
        f"Attention / block     : "
        f"{params['attention_per_block']:,}"
    )

    print(
        f"FFN / block           : "
        f"{params['ffn_per_block']:,}"
    )

    print(
        f"LayerNorm / block     : "
        f"{params['layer_norm_per_block']:,}"
    )

    print(
        f"Transformer block     : "
        f"{params['one_transformer_block']:,}"
    )

    print(
        f"All Transformer       : "
        f"{params['all_transformer_blocks']:,}"
    )

    print(
        f"Final LayerNorm       : "
        f"{params['final_layer_norm']:,}"
    )

    print(
        f"Output bias           : "
        f"{params['output_bias']:,}"
    )

    print("-" * 65)

    print(
        f"TOTAL PARAMETERS      : "
        f"{params['total']:,}"
    )

    print(
        f"TOTAL IN MILLIONS     : "
        f"{params['total'] / 1_000_000:.3f}M"
    )

    print("=" * 65)


if __name__ == "__main__":
    print_parameter_report()