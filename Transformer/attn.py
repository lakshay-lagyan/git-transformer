import numpy as np
from scaled_attention import scaled_attention

class MultiHeadAttn:
    def __init__(self, d_model, num_head):
        assert d_model % num_head == 0

        self.num_head = num_head
        self.d_model = d_model
        self.d_k = d_model // num_head

        self.w_q = np.random.randn(d_model, d_model) * 0.01
        self.w_k = np.random.randn(d_model, d_model) * 0.01
        self.w_v = np.random.randn(d_model, d_model) * 0.01
        self.w_o = np.random.randn(d_model, d_model) * 0.01

    def forward(self, X_q, X_k, X_v, mask=None):
        B, t_q, _ = X_q.shape
        B, t_k, _ = X_k.shape

        Q = X_q @ self.w_q
        K = X_k @ self.w_k
        V = X_v @ self.w_v

        Q = Q.reshape(B, t_q, self.num_head, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B, t_k, self.num_head, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, t_k, self.num_head, self.d_k).transpose(0, 2, 1, 3)

        attn_out, _ = scaled_attention(Q, K, V, mask)  # If your scaled_attention returns two outputs
        concat = attn_out.transpose(0, 2, 1, 3).reshape(B, t_q, self.d_model)

        output = concat @ self.w_o
        return output

d_model, heads = 8, 2
mha = MultiHeadAttn(d_model=d_model, num_head=heads)
X = np.random.randn(1, 5, d_model)
out = mha.forward(X, X, X)