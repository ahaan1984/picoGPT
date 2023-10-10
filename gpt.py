import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import Optional, Tuple

# hyperparameters
batch_size:int = 16 
block_size:int = 32 
max_iters:int = 5000
eval_interval:int = 100
learning_rate:int = 1e-3
device:str = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters:int = 200
n_embd:int = 64
n_head:int = 4
n_layer:int = 4
dropout:int = 0.0

class SingleHeadAttention(nn.Module):
    def __init__(self, head_size:int):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        w = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        w = F.softmax(w, dim=-1) 
        w = self.dropout(w)
        v = self.value(x) 
        out = w @ v 
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads:int, head_size:int):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd:int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), 
            nn.Dropout(dropout)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)
       
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
class Block(nn.Module):
    def __init__(self, n_embd:int, n_head:int):
        super().__init__()  
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)

    def forward(self, x:torch.Tensor) -> torch.Tensor:  
        xres = x                           # store input for residual connection
        x = x + self.sa(self.layernorm1(x))     # apply self attention and layer normalisation
        x = x + self.ffwd(self.layernorm2(x)) # apply linear layer and layer normalisation
        return x + xres                             # apply residual connection after layer normalisation
class GPT(nn.Module):

    def __init__(self, vocab_size:int):
        super().__init__()
        self.token_embd_table = nn.Embedding(vocab_size, n_embd)
        self.position_embd_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )    
        self.layernormf = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
        
    def _init_weights(self, module: torch.nn.Module):
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx:torch.Tensor, targets:Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T,  = idx.shape
        token_embd = self.token_embd_table(idx)
        pos_embd = self.position_embd_table(torch.arange(T, device=device))
        x = token_embd + pos_embd
        x = self.blocks(x)
        x = self.layernormf(x)
        logits = self.lm_head(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, targets)
            return logits, loss
        else: 
            return logits
    
    def generate(self, idx:torch.Tensor, max_new_tokens:int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    


    