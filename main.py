import typing
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2525)

class Bigram:
    def __init__(self, vocab_size, n_embd, n_layer, block_size, n_head) -> torch.Tensor:
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.layernorm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None) -> torch.Tensor:
        B, T = idx.shape
        token_embd = self.token_embedding(idx)
        pos_embd = self.position_embedding(torch.arange(T, device="cpu"))
        x = token_embd + pos_embd
        x = self.blocks(x)
        x = self.layernorm(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


class Block:
    pass


class GELU:
    def forward(self, x) -> int:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))