import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims= True)
    ex= np.exp(x)
    return ex/np.sum(ex,axis=axis, keepdims=True)

def scaled_attention(q, k, v, mask= None):
    d_k = q.shape[-1]
    scores = np.matmul(q, k.transpose(0,1,3,2))/np.sqrt(d_k)
    if mask is not None:
        scores = scores + (mask * - 1e9)
    weight = softmax(scores, axis=-1)
    output = np.matmul(weight, v)
    return output, weight
    
batch, heads, seq, d_k = 1, 1, 3, 4
q = np.random.randn(batch, heads, seq, d_k)
k = np.random.randn(batch, heads, seq, d_k)
v = np.random.randn(batch, heads, seq, d_k)

output, attn_weights = scaled_attention(q, k, v)