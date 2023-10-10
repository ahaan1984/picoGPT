import typing
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size:int = 8 # how many independent sequences will we process in parallel?
block_size:int = 16 # what is the maximum context length for predictions?
max_iters:int = 4000
eval_interval:int = 100
learning_rate:int = 1e-3
device:str = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters:int = 100
n_embd:int = 32
n_head:int = 2
n_layer:int = 2
dropout:int = 0.0


# class Bigram(nn.Module):
#     def __init__(self, vocab_size):
#         super().__init()
#         self.token_embd = nn.Embedding(vocab_size, vocab_size)

#     def forward(self, idx, targets=None) -> torch.Tensor:
#         logits = self.token_embd(idx)
#         if targets is None:
#             loss = None
#         else:
#             b, t, c = logits.shape
#             logits = logits.view(b*t, c)
#             targets = targets.view(b*t)
#             criterion = nn.CrossEntropyLoss()
#             loss = criterion(logits, targets)

#         return logits, loss
    
#     def generate(self, idx, max_new_tokens) -> torch.Tensor:
#         for _ in range(max_new_tokens):
#             logits, loss = self(idx)
#             logits = logits[:, -1, :]
#             probs = F.softmax(logits, dim=-1)
#             idx_next = torch.multinomial(probs, num_samples=1)
#             idx_next = idx_next.to(device)
#             idx = torch.cat((idx, idx_next), dim=1)
#         return idx

class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embd_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embd_table = nn.Embedding(block_size, vocab_size)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.layernormf = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None) -> torch.Tensor:
        b, t = idx.shape
        token_embd = self.token_embd_table(idx)
        pos_embd = self.position_embd_table(torch.arange(t, device=device))
        x = token_embd + pos_embd
        x = self.blocks(x)
        x = self.layernormf(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else: 
            b, t, c = logits.shape
            logits = logits.view(b*t, c)
            targets = targets.view(b*t)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, targets)
        return logits, loss
