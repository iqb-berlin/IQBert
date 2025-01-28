from torch import nn

from bert import datamata
from bert.Settings import Settings


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, settings: Settings):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(settings.d_model, settings.d_ff)
        self.fc2 = nn.Linear(settings.d_ff, settings.d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(datamata.gelu(self.fc1(x)))
