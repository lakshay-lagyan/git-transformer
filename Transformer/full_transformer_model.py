import numpy as np

from pe import positional_encoding
from encoder import encoderLayer
from decoder import Decoder
from embed import Embedding

class Transformer:
    def __init__(self, vocab_size, d_model=32, num_head=4, d_ff=128, num_layer=2, max_len = 100):
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_len, d_model)
        self.enc_layer =[encoderLayer(d_model, num_head, d_ff)
                         for _ in range(num_layer)]
        self.dec_layer = [Decoder(d_model, num_head, d_ff)
                          for _ in range(num_layer)]

        self.w_out = np.random.randn(d_model, vocab_size) * 0.01

    def encoder(self, src_token):
        X = self.embedding.forward(src_token) * np.sqrt(self.d_model)
        X = X + self.pos_encoding[:src_token.shape[1]]
        for layer in self.enc_layer:
            X = layer.forward(X)
        return X
        
    def decoder(self, tgt_token, encoder_output):
        X = self.embedding.forward(tgt_token) * np.sqrt(self.d_model)   
        X = X + self.pos_encoding[:tgt_token.shape[1]]
        seq_len = tgt_token.shape[1]
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        mask = np.tile(mask[None, None, :, :], (X.shape[0], 1, 1, 1))
        for layer in self.dec_layer:
            X= layer.forward(X,encoder_output, self_mask=mask)
        return X
    
    def forward(self, src_token, tgt_token):
        enc_out = self.encoder(src_token)
        dec_out = self.decoder(tgt_token, enc_out)
        B, T, D = dec_out.shape
        logits = dec_out.reshape(-1, D) @ self.w_out
        logits = logits.reshape(B, T, -1)
        return logits
    
vocab = {"<pad>":0, "<s>":1, "The":2, "cat":3, "sat":4, "on":5, "the":6, "mat":7, ".":8}
rev_vocab = {i:w for w, i in vocab.items()}

model = Transformer(vocab_size=len(vocab), d_model=16, num_head=2, d_ff=64, num_layer=1, max_len=10)

prompt = ["<s>", "the", "cat", "sat", "on", "the"]
src_indices = np.array([[vocab[t] for t in prompt]])
tgt_indices = np.array([[vocab["<s>"]]])
logits = model.forward(src_indices, tgt_indices)
probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
next_id = np.argmax(probs[0,-1])

print("predicted new token", rev_vocab[next_id], "(id{})".format(next_id))