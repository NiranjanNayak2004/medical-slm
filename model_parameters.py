import numpy as np


def get_parameters(model):

    parameters = {}

    # ========================================================
    # Embedding
    # ========================================================

    parameters["embedding_matrix"] = (
        model.embedding_matrix
    )

    parameters["output_bias"] = (
        model.output_bias
    )

    # ========================================================
    # Transformer final LayerNorm
    # ========================================================

    parameters["final_gamma"] = (
        model.transformer.gamma
    )

    parameters["final_beta"] = (
        model.transformer.beta
    )

    # ========================================================
    # Transformer blocks
    # ========================================================

    for i, block in enumerate(
        model.transformer.blocks
    ):

        prefix = f"block_{i}"

        parameters[
            f"{prefix}_gamma1"
        ] = block.gamma1

        parameters[
            f"{prefix}_beta1"
        ] = block.beta1

        parameters[
            f"{prefix}_W1"
        ] = block.W1

        parameters[
            f"{prefix}_b1"
        ] = block.b1

        parameters[
            f"{prefix}_W2"
        ] = block.W2

        parameters[
            f"{prefix}_b2"
        ] = block.b2

        parameters[
            f"{prefix}_gamma2"
        ] = block.gamma2

        parameters[
            f"{prefix}_beta2"
        ] = block.beta2

        # Attention parameters

        parameters[
            f"{prefix}_W_Q"
        ] = block.attention.W_Q

        parameters[
            f"{prefix}_W_K"
        ] = block.attention.W_K

        parameters[
            f"{prefix}_W_V"
        ] = block.attention.W_V

        parameters[
            f"{prefix}_W_O"
        ] = block.attention.W_O

        parameters[
            f"{prefix}_b_O"
        ] = block.attention.b_O

    return parameters