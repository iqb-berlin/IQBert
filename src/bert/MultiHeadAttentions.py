from torch import nn, Tensor

from bert.ScaledDotProductAttention import ScaledDotProductAttention
from bert.Settings import Settings


class MultiHeadAttention(nn.Module):  #
    settings: Settings = None

    def __init__(self, settings: Settings):
        super(MultiHeadAttention, self).__init__()
        self.settings = settings
        self.W_Q = nn.Linear(settings.d_model, settings.d_k * settings.n_heads)
        self.W_K = nn.Linear(settings.d_model, settings.d_k * settings.n_heads)
        self.W_V = nn.Linear(settings.d_model, settings.d_v * settings.n_heads)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = query, query.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(query).view(batch_size, -1, self.settings.n_heads, self.settings.d_k).transpose(1, 2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(key).view(batch_size, -1, self.settings.n_heads, self.settings.d_k).transpose(1, 2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(value).view(batch_size, -1, self.settings.n_heads, self.settings.d_v).transpose(1, 2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.settings.n_heads, 1, 1)  # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention(self.settings)(q_s, k_s, v_s, attention_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.settings.n_heads * self.settings.d_v)  # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(self.settings.n_heads * self.settings.d_v, self.settings.d_model)(context)
        return nn.LayerNorm(self.settings.d_model)(output + residual), attn  # output: [batch_size x len_q x d_model]
