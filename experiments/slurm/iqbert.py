from typing import Union

from datasets import Dataset, load_dataset, DatasetDict, IterableDatasetDict, IterableDataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, \
    PreTrainedTokenizerBase

from read_data import partition


def load_data() -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    # data = {"text": [...], "label": [...]}
    # dataset = Dataset.from_dict(data)
    return load_dataset("imdb")

def tokenize_data(tokenizer: PreTrainedTokenizerBase, dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> DatasetDict:
    return dataset.map(
        lambda item: tokenizer(item["text"], truncation=True, padding="max_length"),
        batched=True
    )

def finetune_model(dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]):
    dataset = dataset.train_test_split(test=0.1)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_dataset = tokenize_data(tokenizer, dataset)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    trainer.train()

    trainer.save_model("./bert-finetuned")
    tokenizer.save_pretrained("./bert-finetuned")

