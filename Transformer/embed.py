import numpy as np
class Embedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = np.random.randn(vocab_size, d_model) * 0.01

    def forward(self, X):
        return self.weight[X]
    
        
vocab = {"<s>":0, "The":1, "cat":2, "sat":3, "on":4, "the":5, ".":6}
batch_token = np.array([[vocab["The"], vocab["cat"], vocab["sat"]]])
embed_layer = Embedding(vocab_size=len(vocab), d_model=8)
embedded = embed_layer.forward(batch_token)