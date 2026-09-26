import numpy as np

from model_config import EMBEDDING_DIM, FFN_DIM
from multi_head_attention_v2 import MultiHeadAttention


class TransformerBlock:

    def __init__(self):

        self.attention = MultiHeadAttention()

        # LayerNorm 1
        self.gamma1 = np.ones(EMBEDDING_DIM)
        self.beta1 = np.zeros(EMBEDDING_DIM)

        # Feed Forward Network
        self.W1 = (
            np.random.randn(
                EMBEDDING_DIM,
                FFN_DIM
            ) * 0.02
        )

        self.b1 = np.zeros(FFN_DIM)

        self.W2 = (
            np.random.randn(
                FFN_DIM,
                EMBEDDING_DIM
            ) * 0.02
        )

        self.b2 = np.zeros(EMBEDDING_DIM)

        # LayerNorm 2
        self.gamma2 = np.ones(EMBEDDING_DIM)
        self.beta2 = np.zeros(EMBEDDING_DIM)

        self.cache = {}


    # ========================================================
    # LayerNorm
    # ========================================================

    def layer_norm(self, x, gamma, beta):

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
            (x - mean) / std
        )

        output = (
            gamma * normalized
            + beta
        )

        return output, (
            x,
            normalized,
            mean,
            std
        )


    # ========================================================
    # LayerNorm backward
    # ========================================================

    def layer_norm_backward(
        self,
        d_output,
        cache,
        gamma
    ):

        x, normalized, mean, std = cache

        sequence_length = x.shape[0]
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
            d_output * gamma
        )

        x_centered = x - mean

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

        return d_x, d_gamma, d_beta


    # ========================================================
    # Forward
    # ========================================================

    def forward(self, X):

        # ------------------------------------
        # Attention
        # ------------------------------------

        attention_output = (
            self.attention.forward(X)
        )

        # First residual connection
        residual1 = (
            X + attention_output
        )

        # LayerNorm 1
        norm1, norm1_cache = (
            self.layer_norm(
                residual1,
                self.gamma1,
                self.beta1
            )
        )

        # ------------------------------------
        # Feed Forward Network
        # ------------------------------------

        hidden = (
            norm1 @ self.W1
            + self.b1
        )

        relu_output = np.maximum(
            0,
            hidden
        )

        ffn_output = (
            relu_output @ self.W2
            + self.b2
        )

        # Second residual connection
        residual2 = (
            norm1 + ffn_output
        )

        # LayerNorm 2
        output, norm2_cache = (
            self.layer_norm(
                residual2,
                self.gamma2,
                self.beta2
            )
        )

        # Cache everything needed by backward
        self.cache = {
            "X": X,
            "residual1": residual1,
            "norm1": norm1,
            "hidden": hidden,
            "relu_output": relu_output,
            "ffn_output": ffn_output,
            "residual2": residual2,
            "norm1_cache": norm1_cache,
            "norm2_cache": norm2_cache
        }

        return output


    # ========================================================
    # Backward
    # ========================================================

    def backward(self, d_output):

        X = self.cache["X"]
        norm1 = self.cache["norm1"]
        hidden = self.cache["hidden"]
        relu_output = self.cache["relu_output"]

        # ====================================
        # LayerNorm 2 backward
        # ====================================

        d_residual2, d_gamma2, d_beta2 = (
            self.layer_norm_backward(
                d_output,
                self.cache["norm2_cache"],
                self.gamma2
            )
        )

        # residual2 = norm1 + ffn_output
        d_norm1_from_residual = (
            d_residual2
        )

        d_ffn_output = (
            d_residual2
        )

        # ====================================
        # FFN backward
        # ====================================

        d_W2 = (
            relu_output.T @ d_ffn_output
        )

        d_b2 = np.sum(
            d_ffn_output,
            axis=0
        )

        d_relu = (
            d_ffn_output @ self.W2.T
        )

        # ReLU derivative
        d_hidden = (
            d_relu *
            (hidden > 0)
        )

        d_W1 = (
            norm1.T @ d_hidden
        )

        d_b1 = np.sum(
            d_hidden,
            axis=0
        )

        d_norm1_from_ffn = (
            d_hidden @ self.W1.T
        )

        # Combine gradients entering norm1
        d_norm1 = (
            d_norm1_from_residual
            + d_norm1_from_ffn
        )

        # ====================================
        # LayerNorm 1 backward
        # ====================================

        d_residual1, d_gamma1, d_beta1 = (
            self.layer_norm_backward(
                d_norm1,
                self.cache["norm1_cache"],
                self.gamma1
            )
        )

        # residual1 = X + attention_output
        d_X_residual = (
            d_residual1
        )

        d_attention_output = (
            d_residual1
        )

        # ====================================
        # Attention backward
        # ====================================

        d_X_attention, attention_grads = (
            self.attention.backward(
                d_attention_output
            )
        )

        # ====================================
        # Residual gradient
        # ====================================

        d_X = (
            d_X_residual
            + d_X_attention
        )

        gradients = {

            "gamma1": d_gamma1,
            "beta1": d_beta1,

            "W1": d_W1,
            "b1": d_b1,

            "W2": d_W2,
            "b2": d_b2,

            "gamma2": d_gamma2,
            "beta2": d_beta2,

            "attention": attention_grads
        }

        return d_X, gradients