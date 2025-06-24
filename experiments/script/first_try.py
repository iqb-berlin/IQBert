import torch
from torch import nn, optim

import bert.processor
import bert.datamata
from bert.Bert import BERT
from bert.Settings import Settings

text = (
    'Hello, how are you? I am Romeo.\n'
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'
    'Nice meet you too. How are you today?\n'
    'Great. My baseball team won the competition.\n'
    'Oh Congratulations, Juliet\n'
    'Thanks you Romeo'
)

settings = Settings()

prepared_text = bert.processor.prepare(text)


model = BERT(prepared_text.vocab_size, settings)

batch = bert.processor.make_batch(prepared_text, settings.maxlen, settings.batch_size, settings.max_pred)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))

# -- training

for epoch in range(100):
    optimizer.zero_grad()
    logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
    loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens) # for masked LM
    loss_lm = (loss_lm.float()).mean()
    loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
    loss = loss_lm + loss_clsf
    if (epoch + 1) % 10 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()


# -- output

print(text)
print([prepared_text.number_dict[w.item()] for w in input_ids[0] if prepared_text.number_dict[w.item()] != '[PAD]'])

logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos) # Logits are typically the raw, unnormalized predictions (scores) for each token in your vocabulary
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('masked tokens list : ', [pos.item() for pos in masked_tokens[0] if pos.item() != 0])
print('predict masked tokens list : ', [pos for pos in logits_lm if pos != 0])

print('predicted words for masked tokens : ', [pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext : ', isNext) # This is a tensor showing whether the model predicts the next sentence as a continuation of the current one or not.
print('predict isNext : ', True if logits_clsf else False) # This suggests that the model's final prediction is that the second sentence in the input is likely the continuation of the first sentence.

