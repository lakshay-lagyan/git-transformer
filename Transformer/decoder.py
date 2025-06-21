import numpy as np

from fnn import Feedforward
from L_normal import LayerNorm
from attn import MultiHeadAttn


class Decoder:
    def __init__(self, d_model, num_head, d_ff):
        self.attn = MultiHeadAttn(d_model, num_head)
        self.dec_attn = MultiHeadAttn(d_model, num_head)
        self.ffn = Feedforward(d_model, d_ff)
        self.norma1 = LayerNorm(d_model)
        self.norma2 = LayerNorm(d_model)
        self.norma3 = LayerNorm(d_model)

    def forward(self, X, enc_layer, self_mask=None, enc_dec_mask = None):

        attn1 = self.attn.forward(X, X, X, self_mask)
        X = self.norma1.forward(X + attn1)

        attn2 = self.dec_attn.forward(X, enc_layer, enc_layer, enc_dec_mask)
        X = self.norma2.forward(X + attn2)

        ffn_out = self.ffn.forward(X)
        X = self.norma3.forward(X + ffn_out)
        return X
    
decoder_state  = Decoder(d_model=16, num_head=2, d_ff=64)
dec_inp = np.random.randn(1, 4, 16)
enc_out = np.random.randn(1, 5, 16)

output = decoder_state.forward(dec_inp, enc_out)
