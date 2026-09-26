def get_gradients(model, gradients):

    result = {}

    # ========================================================
    # Embedding
    # ========================================================

    result["embedding_matrix"] = (
        gradients["embedding_matrix"]
    )

    result["output_bias"] = (
        gradients["output_bias"]
    )

    # ========================================================
    # Final LayerNorm
    # ========================================================

    result["final_gamma"] = (
        gradients["transformer"]["gamma"]
    )

    result["final_beta"] = (
        gradients["transformer"]["beta"]
    )

    # ========================================================
    # Transformer blocks
    # ========================================================

    for i, block_gradient in enumerate(
        gradients["transformer"]["blocks"]
    ):

        prefix = f"block_{i}"

        result[
            f"{prefix}_gamma1"
        ] = block_gradient["gamma1"]

        result[
            f"{prefix}_beta1"
        ] = block_gradient["beta1"]

        result[
            f"{prefix}_W1"
        ] = block_gradient["W1"]

        result[
            f"{prefix}_b1"
        ] = block_gradient["b1"]

        result[
            f"{prefix}_W2"
        ] = block_gradient["W2"]

        result[
            f"{prefix}_b2"
        ] = block_gradient["b2"]

        result[
            f"{prefix}_gamma2"
        ] = block_gradient["gamma2"]

        result[
            f"{prefix}_beta2"
        ] = block_gradient["beta2"]

        attention = block_gradient[
            "attention"
        ]

        result[
            f"{prefix}_W_Q"
        ] = attention["W_Q"]

        result[
            f"{prefix}_W_K"
        ] = attention["W_K"]

        result[
            f"{prefix}_W_V"
        ] = attention["W_V"]

        result[
            f"{prefix}_W_O"
        ] = attention["W_O"]

        result[
            f"{prefix}_b_O"
        ] = attention["b_O"]

    return result