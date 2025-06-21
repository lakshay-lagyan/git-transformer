import numpy as np

from fnn import Feedforward
from L_normal import LayerNorm
from attn import MultiHeadAttn

class encoderLayer:
    def __init__(self, d_model, num_head, d_ff):
        self.attn = MultiHeadAttn(d_model, num_head)
        self.ffn = Feedforward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, X, mask = None):
        multiAttn = self.attn.forward(X, X, X, mask)
        X = self.norm1.forward(X + multiAttn)

        ffn_out = self.ffn.forward(X)
        X = self.norm2.forward(X + ffn_out)

        return X

enc_block = encoderLayer(d_model=16, num_head=2, d_ff=64)
X = np.random.randn(1, 5 ,16)
out = enc_block.forward(X)
    