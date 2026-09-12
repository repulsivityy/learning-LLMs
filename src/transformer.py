import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    output = weights @ V
    return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        output, weights = scaled_dot_product_attention(Q, K, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.W_o(output), weights


class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = ResidualLayerNorm(d_model)  # First for Attention
        self.ff = FeedForward(d_model, d_ff)
        self.add_norm2 = ResidualLayerNorm(d_model)  # Next for Feed-Forward

    def forward(self, x):
        attn_output, weights = self.mha(x)
        x = self.add_norm1(x, attn_output)

        ff_output = self.ff(x)
        x = self.add_norm2(x, ff_output)

        return x, weights


class MultiHeadAttentionWithCache(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.reset_cache()

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.kv_computations = 0  # how many tokens' K/V we've EVER computed

    def _split_heads(self, x, batch_size, seq_len):
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def step(self, x_new):
        batch_size = x_new.shape[0]

        Q = self._split_heads(self.W_q(x_new), batch_size, 1)
        K_new = self._split_heads(self.W_k(x_new), batch_size, 1)
        V_new = self._split_heads(self.W_v(x_new), batch_size, 1)
        self.kv_computations += 1

        if self.k_cache is None:
            self.k_cache, self.v_cache = K_new, V_new
        else:
            self.k_cache = torch.cat([self.k_cache, K_new], dim=2)
            self.v_cache = torch.cat([self.v_cache, V_new], dim=2)

        output, weights = scaled_dot_product_attention(Q, self.k_cache, self.v_cache)
        output = output.transpose(1, 2).contiguous().view(batch_size, 1, -1)
        return self.W_o(output), weights
