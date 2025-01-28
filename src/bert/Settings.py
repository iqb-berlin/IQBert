from dataclasses import dataclass

@dataclass
class Settings:
    maxlen: int = 30 # maximum of length
    batch_size: int = 6
    max_pred: int = 5  # max tokens of prediction
    n_layers: int = 6 # number of Encoder Layer
    n_heads: int = 12 # number of heads in Multi-Head Attention
    d_model: int = 768 # Embedding Size
    d_ff: int = 768 * 4  # 4*d_model, FeedForward dimension
    d_k: int = 64  # dimension of K(=Q), V
    d_v: int = 64
    n_segments: int = 2
