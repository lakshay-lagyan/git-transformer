import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps = 1e-6):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta  = np.zeros(d_model)

    def forward(self, X):
        mean = np.mean(X, axis=-1, keepdims=True)
        var  = np.var(X, axis=-1, keepdims=True)
        X_norm = (X - mean) / np.sqrt(var * self.eps)
        return self.gamma * X_norm + self.beta    