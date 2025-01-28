import re
from random import randrange, random, randint, shuffle
from bert.PreparedText import PreparedText
from bert.Batch import Batch


def prepare(text: str) -> PreparedText:
    sentences = get_sentences(text)
    word_dict, number_dict = _get_dictionary(sentences)
    vocab_size = len(word_dict)
    token_list = _get_token_list(sentences, word_dict)
    return PreparedText(
        sentences=sentences,
        vocab_size=vocab_size,
        token_list=token_list,
        word_dict=word_dict,
        number_dict=number_dict
    )

def get_sentences(text: str) -> list[str]:
    return re.sub("[.,!?\\-]", '', text.lower()).split('\n') # TODO why split onto \n

def _get_dictionary(sentences: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    word_list = list(set(" ".join(sentences).split()))
    word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i, w in enumerate(word_list):
        word_dict[w] = i + 4
    number_dict = {i: w for i, w in enumerate(word_dict)}
    return word_dict, number_dict

def _get_token_list(sentences: list[str], word_dict: dict[str, int]) -> list[list[int]]:
    token_list = list()
    for sentence in sentences:
        arr = [word_dict[s] for s in sentence.split()]
        token_list.append(arr)
    return token_list

def make_batch(text: PreparedText, maxlen: int, batch_size: int, max_pred: int) -> Batch:
    batch = []
    positive = negative = 0
    while positive != batch_size / 2 or negative != batch_size / 2:
        tokens_a_index, tokens_b_index = randrange(len(text.sentences)), randrange(len(text.sentences))

        tokens_a, tokens_b = text.token_list[tokens_a_index], text.token_list[tokens_b_index] # TODO rename to sentence_a and sentence_b ?

        # wir nehmen zwei zufällige (codierte) Sätze packen sie hintereinander, dazwischen SEP bzw. CLS
        input_ids = [text.word_dict['[CLS]']] + tokens_a + [text.word_dict['[SEP]']] + tokens_b + [text.word_dict['[SEP]']]
        # liste zu welchem segment (satz) die Stelle jeweils gehört
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        n_pred =  min(max_pred, max(1, int(round(len(input_ids) * 0.15)))) # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids) if token != text.word_dict['[CLS]'] and token != text.word_dict['[SEP]']]
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = text.word_dict['[MASK]'] # make mask
            elif random() < 0.5:  # 10%
                index = randint(0, text.vocab_size - 1) # random index in vocabulary
                input_ids[pos] = text.word_dict[text.number_dict[index]] # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        # TODO True/False soll sagen, ob Kontext (satz b folgt auf a) - aber wird nicht das Gegenteil abgeprüft?
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            negative += 1
    return batch
