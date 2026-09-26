import numpy as np


class AdamOptimizer:

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.v = {}

        self.step_count = 0

    def update(self, parameters, gradients):

        self.step_count += 1

        updated_parameters = {}

        for name in parameters:

            parameter = parameters[name]
            gradient = gradients[name]

            # Initialize moment estimates
            if name not in self.m:

                self.m[name] = np.zeros_like(
                    parameter
                )

                self.v[name] = np.zeros_like(
                    parameter
                )

            # First moment
            self.m[name] = (
                self.beta1 * self.m[name]
                +
                (1 - self.beta1) * gradient
            )

            # Second moment
            self.v[name] = (
                self.beta2 * self.v[name]
                +
                (1 - self.beta2) * gradient ** 2
            )

            # Bias correction
            m_corrected = (
                self.m[name]
                /
                (
                    1 -
                    self.beta1 ** self.step_count
                )
            )

            v_corrected = (
                self.v[name]
                /
                (
                    1 -
                    self.beta2 ** self.step_count
                )
            )

            # Parameter update
            parameter = (
                parameter
                -
                self.learning_rate
                *
                m_corrected
                /
                (
                    np.sqrt(v_corrected)
                    +
                    self.epsilon
                )
            )

            updated_parameters[name] = parameter

        return updated_parameters