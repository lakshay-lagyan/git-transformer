import numpy as np

class Feedforward:
    def __init__(self, d_model, d_ff):
        self.w1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.w2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros(d_model)

    def forward(self, X):
        hidden = np.dot(X, self.w1) + self.b1
        hidden = np.maximum(0, hidden)

        output = np.dot(hidden, self.w2) + self.b2
        return output
