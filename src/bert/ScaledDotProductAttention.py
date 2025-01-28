from typing import Any
import torch
from torch import nn, Tensor
import numpy as np

from bert.Settings import Settings


class ScaledDotProductAttention(nn.Module):
    def __init__(self, settings: Settings):
        super(ScaledDotProductAttention, self).__init__()
        self.settings = settings

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_mask) -> [Tensor, Any]:
        scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(self.settings.d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attention_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, value)
        return context, attn
