from torch import nn

from bert.MultiHeadAttentions import MultiHeadAttention
from bert.PoswiseFeedForwardNet import PoswiseFeedForwardNet
from bert.Settings import Settings


class EncoderLayer(nn.Module):
    def __init__(self, settings: Settings):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(settings)
        self.pos_ffn = PoswiseFeedForwardNet(settings)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn
