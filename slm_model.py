import numpy as np
from transformer import TransformerBlock


class MedicalSLM:

    def __init__(self, vocabulary_size, embedding_size):

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        # Token embeddings
        self.embedding_matrix = np.random.randn(
            vocabulary_size,
            embedding_size
        ) * 0.1

        # Transformer
        self.transformer = TransformerBlock(
            embedding_size
        )

        # Output layer
        self.output_weights = np.random.randn(
            embedding_size,
            vocabulary_size
        ) * 0.1

        self.output_bias = np.zeros(
            vocabulary_size
        )


    def softmax(self, x):

        exp_x = np.exp(
            x - np.max(x)
        )

        return exp_x / np.sum(exp_x)


    def forward(self, input_ids):

        # 1. Convert token IDs to embeddings

        token_embeddings = self.embedding_matrix[
            input_ids
        ]

        # Example:
        # input_ids = [2, 5, 7]
        #
        # token_embeddings =
        # [
        #   embedding of token 2
        #   embedding of token 5
        #   embedding of token 7
        # ]


        # 2. Pass embeddings through Transformer

        transformer_output = self.transformer.forward(
            token_embeddings
        )


        # 3. Use the LAST token representation

        last_token = transformer_output[-1]


        # 4. Convert representation into vocabulary scores

        logits = np.dot(
            last_token,
            self.output_weights
        ) + self.output_bias


        # 5. Convert scores into probabilities

        probabilities = self.softmax(
            logits
        )


        return logits, probabilities


    def predict_next_word(
        self,
        input_ids,
        id_to_word
    ):

        logits, probabilities = self.forward(
            input_ids
        )

        predicted_id = np.argmax(
            probabilities
        )

        predicted_word = id_to_word[
            predicted_id
        ]

        return predicted_word, probabilities
if __name__ == "__main__":

    vocabulary = {
        "fever": 0,
        "is": 1,
        "a": 2,
        "common": 3,
        "symptom": 4
    }

    id_to_word = {
        value: key
        for key, value in vocabulary.items()
    }

    model = MedicalSLM(
        vocabulary_size=len(vocabulary),
        embedding_size=4
    )

    input_ids = np.array([
        vocabulary["fever"],
        vocabulary["is"],
        vocabulary["a"]
    ])

    predicted_word, probabilities = model.predict_next_word(
        input_ids,
        id_to_word
    )

    print("Predicted word:")
    print(predicted_word)

    print("\nProbabilities:")
    print(probabilities)