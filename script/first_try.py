import bert.processor
import bert.datamata
from bert.Embedding import Embedding
from bert.ScaledDotProductAttention import ScaledDotProductAttention

text = (
    'Hello, how are you? I am Romeo.\n'
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'
    'Nice meet you too. How are you today?\n'
    'Great. My baseball team won the competition.\n'
    'Oh Congratulations, Juliet\n'
    'Thanks you Romeo'
)

# settings TODO put in class
maxlen = 30 # maximum of length
batch_size = 6
max_pred = 5  # max tokens of prediction
n_layers = 6 # number of Encoder of Encoder Layer
n_heads = 12 # number of heads in Multi-Head Attention
d_model = 768 # Embedding Size
d_ff = 768 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2
# settings over

prepared_text = bert.processor.prepare(text)
batch = bert.processor.make_batch(prepared_text, maxlen, batch_size, max_pred)

input_ids, segment_ids, masked_tokens, masked_pos, isNext = bert.datamata.do_the_shit(batch)

emb = Embedding(maxlen, d_model, n_segments)
embeds = emb(input_ids, segment_ids)

attenM = bert.datamata.get_attn_pad_mask(input_ids, input_ids)

SDPA = ScaledDotProductAttention()(embeds, embeds, embeds, attenM)

S, C, A = SDPA


print('Scores: ', S[0][0],'\n\nAttention M: ', A[0][0])




