import math
import torch
import numpy
from torch import nn
from torch.autograd.grad_mode import F


class LanguageModel(nn.Module):
    def __init__(self, dim_model, dim_ff, vocab_size, num_layers, num_head, dropout):
        super(LanguageModel, self).__init__()
        self.embd = Embeddings(dim_model, vocab_size)
        self.pos_encoding = PositionalEncoding(dim_model, dropout)
        self.decoder = Decoder(dim_model, num_layers, num_head, dim_ff, dropout)
        self.output_embd = OutputEmbeddings(dim_model, vocab)

    def forward(self, x, mask):
        x = self.embd(x)
        x = self.pos_encoding(x)
        x = self.decoder(x, mask)
        x = self.output_embd(x)

        return x

class Embeddings(nn.Module):
    def __init__(self, dim_model, vocab_size):
        super(Embeddings, self).__init__()
        self.embd_layer = nn.Embedding(vocab_size, dim_model)
        self.dim_model = dim_model

    def forward(self, x):
        return self.embd_layer(x) * math.sqrt(self.dim_model)

class PositionalEncoding():
    def __init__(self, dim_model, dropout):
        super(PositionalEncoding, self).__init__()
        self.dim_model = dim_model
        self.dropout = nn.Dropout(dropout)
        self.pe = torch.zeros(5000, dim_model)
        for pos in range(5000):
            for i in range(dim_model):
                if i % 2 == 0:
                    self.pe[pos, i] = math.sin(pos / math.pow(10000, (2.0*i)/dim_model))
                else:
                    self.pe[pos, i] = math.cos(pos / math.pow(10000, (2.0*i)/dim_model))

    def forward(self, x): #0 batch_size, 1 seq_len, 2, dim_model, pe, 0 5000, 1, dim_model
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x

def main():
    position = PositionalEncoding(1024, 0.1)
    print(position.pe[:10, :4])

if __name__ == "__main__":
    main()




def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
