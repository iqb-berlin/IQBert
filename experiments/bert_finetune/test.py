from datasets import load_dataset

from iqbert import finetune_model

data = load_dataset("imdb")
finetune_model(data)
