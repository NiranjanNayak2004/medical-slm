import numpy as np

from model_config import NUM_LAYERS, EMBEDDING_DIM
from transformer_block_v3 import TransformerBlock


class TransformerModel:

    def __init__(self):

        self.blocks = [
            TransformerBlock()
            for _ in range(NUM_LAYERS)
        ]

        # Final LayerNorm
        self.gamma = np.ones(EMBEDDING_DIM)
        self.beta = np.zeros(EMBEDDING_DIM)

        self.cache = None

    # ========================================================
    # LayerNorm
    # ========================================================

    def layer_norm(self, x):

        mean = np.mean(
            x,
            axis=-1,
            keepdims=True
        )

        variance = np.var(
            x,
            axis=-1,
            keepdims=True
        )

        std = np.sqrt(
            variance + 1e-5
        )

        normalized = (
            x - mean
        ) / std

        output = (
            self.gamma * normalized
            + self.beta
        )

        self.cache = (
            x,
            normalized,
            mean,
            std
        )

        return output

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, X):

        hidden = X

        for block in self.blocks:

            hidden = block.forward(
                hidden
            )

        hidden = self.layer_norm(
            hidden
        )

        return hidden

    # ========================================================
    # LayerNorm backward
    # ========================================================

    def layer_norm_backward(
        self,
        d_output
    ):

        x, normalized, mean, std = (
            self.cache
        )

        dimension = x.shape[1]

        d_gamma = np.sum(
            d_output * normalized,
            axis=0
        )

        d_beta = np.sum(
            d_output,
            axis=0
        )

        d_normalized = (
            d_output * self.gamma
        )

        d_x = (
            1.0 / dimension
        ) * (
            1.0 / std
        ) * (
            dimension * d_normalized
            - np.sum(
                d_normalized,
                axis=1,
                keepdims=True
            )
            - normalized
            * np.sum(
                d_normalized * normalized,
                axis=1,
                keepdims=True
            )
        )

        return (
            d_x,
            d_gamma,
            d_beta
        )

    # ========================================================
    # Backward
    # ========================================================

    def backward(self, d_output):

        # ------------------------------------
        # Final LayerNorm
        # ------------------------------------

        d_hidden, d_gamma, d_beta = (
            self.layer_norm_backward(
                d_output
            )
        )

        block_gradients = []

        # ------------------------------------
        # Transformer blocks
        # Reverse order
        # ------------------------------------

        for block in reversed(
            self.blocks
        ):

            d_hidden, gradients = (
                block.backward(
                    d_hidden
                )
            )

            block_gradients.append(
                gradients
            )

        # Put gradients back
        # into normal block order

        block_gradients.reverse()

        return d_hidden, {
            "blocks": block_gradients,
            "gamma": d_gamma,
            "beta": d_beta
        }