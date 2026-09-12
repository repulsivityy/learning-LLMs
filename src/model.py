import torch
import torch.nn as nn

from transformer import causal_mask, TransformerBlock


class TokenAndPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        return self.token_embedding(token_ids) + self.position_embedding(positions)


class ToyLanguageModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.embedding = TokenAndPositionalEmbedding(vocab_size, max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        x = self.embedding(token_ids)
        mask = causal_mask(seq_len)

        attn_weights_per_layer = []
        for block in self.blocks:
            x, weights = block(x, mask=mask)
            attn_weights_per_layer.append(weights)

        logits = self.output_head(x)
        return logits, attn_weights_per_layer
