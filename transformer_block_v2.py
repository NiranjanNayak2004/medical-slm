import numpy as np

from model_config import EMBEDDING_DIM, FFN_DIM
from multi_head_attention import MultiHeadAttention


class TransformerBlock:

    def __init__(self):

        self.attention = MultiHeadAttention()

        # LayerNorm 1
        self.gamma1 = np.ones(EMBEDDING_DIM)
        self.beta1 = np.zeros(EMBEDDING_DIM)

        # Feed Forward Network
        self.W1 = np.random.randn(
            EMBEDDING_DIM,
            FFN_DIM
        ) * 0.02

        self.b1 = np.zeros(FFN_DIM)

        self.W2 = np.random.randn(
            FFN_DIM,
            EMBEDDING_DIM
        ) * 0.02

        self.b2 = np.zeros(EMBEDDING_DIM)

        # LayerNorm 2
        self.gamma2 = np.ones(EMBEDDING_DIM)
        self.beta2 = np.zeros(EMBEDDING_DIM)

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

        normalized = (
            x - mean
        ) / np.sqrt(
            variance + 1e-5
        )

        return gamma * normalized + beta

    def relu(self, x):
        return np.maximum(0, x)

    def feed_forward(self, x):

        hidden = x @ self.W1 + self.b1

        hidden = self.relu(hidden)

        output = hidden @ self.W2 + self.b2

        return output

    def forward(self, X):

        # -----------------------------------------
        # 1. Multi-Head Self Attention
        # -----------------------------------------

        attention_output = self.attention.forward(X)

        # -----------------------------------------
        # 2. Residual connection
        # -----------------------------------------

        x = X + attention_output

        # -----------------------------------------
        # 3. Layer Normalization
        # -----------------------------------------

        x = self.layer_norm(
            x,
            self.gamma1,
            self.beta1
        )

        # -----------------------------------------
        # 4. Feed Forward Network
        # -----------------------------------------

        ffn_output = self.feed_forward(x)

        # -----------------------------------------
        # 5. Second residual connection
        # -----------------------------------------

        x = x + ffn_output

        # -----------------------------------------
        # 6. Second Layer Normalization
        # -----------------------------------------

        x = self.layer_norm(
            x,
            self.gamma2,
            self.beta2
        )

        return x


if __name__ == "__main__":

    print("=" * 60)
    print("        Transformer Block Test")
    print("=" * 60)

    sequence_length = 128

    X = np.random.randn(
        sequence_length,
        EMBEDDING_DIM
    )

    transformer = TransformerBlock()

    output = transformer.forward(X)

    print()
    print("Input shape       :", X.shape)
    print("Attention output  :", output.shape)
    print("FFN hidden size   :", FFN_DIM)
    print("Final output      :", output.shape)

    print()
    print("Transformer block working correctly.")