import math

import torch
from torch import Tensor

import bert
from bert.Batch import Batch
from bert.Embedding import Embedding
from bert.MultiHeadAttentions import MultiHeadAttention
from bert.PreparedText import PreparedText
from bert.ScaledDotProductAttention import ScaledDotProductAttention
from bert.Settings import Settings


def do_the_shit(batch: list[list[list[str | int] | list[int] | bool]]):
    return map(torch.LongTensor, zip(*batch))

def get_attn_pad_mask(seq_q, seq_k) -> Tensor:
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def gelu(x: float) -> Tensor: # Gaussian Error Linear Unit
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def do_it_with_sdpa(prepared_text: PreparedText, batch: Batch, settings: Settings):
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = do_the_shit(batch)
    emb = Embedding(prepared_text.vocab_size, settings)
    embeds = emb(input_ids, segment_ids)

    attenM = get_attn_pad_mask(input_ids, input_ids)

    SDPA = ScaledDotProductAttention(settings)(embeds, embeds, embeds, attenM)
    print(SDPA)
    scores, attention = SDPA
    print('Scores: ', scores[0][0],'\n\nAttention M: ', attention[0][0])


def do_it_with_mha(prepared_text: PreparedText, batch: Batch, settings: Settings):
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = do_the_shit(batch)
    emb = Embedding(prepared_text.vocab_size, settings)
    embeds = emb(input_ids, segment_ids)

    attenM = get_attn_pad_mask(input_ids, input_ids)

    MHA = MultiHeadAttention(settings)(embeds, embeds, embeds, attenM)

    Output, A = MHA
    print(Output)
