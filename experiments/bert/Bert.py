import torch
from torch import nn

from bert import datamata
from bert.Embedding import Embedding
from bert.EncoderLayer import EncoderLayer
from bert.Settings import Settings


class BERT(nn.Module):
    def __init__(self, vocab_size: int, settings: Settings):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, settings)
        self.layers = nn.ModuleList([EncoderLayer(settings) for _ in range(settings.n_layers)])
        self.fc = nn.Linear(settings.d_model, settings.d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(settings.d_model, settings.d_model)
        self.activ2 = datamata.gelu
        self.norm = nn.LayerNorm(settings.d_model)
        self.classifier = nn.Linear(settings.d_model, 2)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = datamata.get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        # it will be decided by first token(CLS)
        h_pooled = self.activ1(self.fc(output[:, 0])) # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2]

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]
        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_clsf
