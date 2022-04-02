import math
import torch
import numpy
from torch.autograd.grad_mode import F


class LanguageModel():
    def __init__(self, dim_model, dim_ff, vocab, num_layers, num_head, dropout):
        self.embd = Embeddings(dim_model, vocab)
        self.pos_encoding = PositionalEncoding(dim_model, dropout)
        self.decoder = Decoder(dim_model, num_layers, num_head, dim_ff, dropout)
        self.output_embd = OutputEmbeddings(dim_model, vocab)

    def forward(self, x, mask):
        x = self.embd(x)
        x = self.pos_encoding(x)
        x = self.decoder(x, mask)
        x = self.output_embd(x)

        return x

class Embeddings():
    def __init__(self):

class





def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
