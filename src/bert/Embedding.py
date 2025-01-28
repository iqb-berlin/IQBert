import torch
from torch import nn

from bert.Settings import Settings


class Embedding(nn.Module):
    settings: Settings = None

    def __init__(self, vocab_size: int, settings: Settings):
        super(Embedding, self).__init__()
        self.settings = settings
        # vocab_size tensors of size d_model
        self.tok_embed = nn.Embedding(vocab_size, self.settings.d_model)  # token embedding
        self.pos_embed = nn.Embedding(self.settings.maxlen, self.settings.d_model)  # position embedding
        self.seg_embed = nn.Embedding(self.settings.n_segments, self.settings.d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(self.settings.d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)
