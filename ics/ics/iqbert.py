from typing import Union, Callable

from datasets import Dataset, DatasetDict, IterableDatasetDict, IterableDataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, \
    PreTrainedTokenizerBase, TrainerCallback


def tokenize_data(tokenizer: PreTrainedTokenizerBase, dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> DatasetDict:
    return dataset.map(
        lambda item: tokenizer(item["text"], truncation=True, padding="max_length"),
        batched=True
    )

def finetune_model(
    dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
    name: str,
    reporter: Callable[[str, bool], None]
):
    dataset = dataset.train_test_split(test_size=0.1)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_dataset = tokenize_data(tokenizer, dataset)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # TODO a validation dataset and add load_best_model_at_end
    # TODO add optional seed= for reproducibility

    # TODO experiment with a learning rate scheduler (warmup)
    # TODO optimize parameters
    training_args = TrainingArguments(
        output_dir="./results",
        save_strategy="epoch",
        per_device_train_batch_size=8, # not too high since datasets are quite small and we want to avoid overfitting
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.05, # how we penal extraordinary weighty tokens. keep small for small datasets
        learning_rate=2e-5 # don't destroy what bert already have learned
    )

    class CustomCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            print(f"Epoch {state.epoch} ended!")
            reporter(f"{state.epoch}/{args.num_train_epochs}", False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        callbacks=[CustomCallback]
    )

    trainer.train()

    trainer.save_model(f"./data/{name}")
    tokenizer.save_pretrained(f"./data/{name}")

