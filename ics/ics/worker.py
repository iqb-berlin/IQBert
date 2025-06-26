import os
import uuid
from typing import List, Callable

import torch
from datasets import Dataset
from redis import StrictRedis

from ics_components.common.models import TrainingResult
from ics.implementation import TaskInstructions, CoderRegistry
from ics_models import Response, Code
from transformers import BertTokenizer, BertForSequenceClassification

from .iqbert import finetune_model

redis_host = os.getenv('REDIS_HOST') or 'localhost'
redis_store = StrictRedis(host=redis_host, port=6379, db=0, decode_responses=True)

def to_dataset(data: list[Response]) -> Dataset:
    texts = []
    labels = []
    for item in data:
        texts.append(item["value"])
        labels.append(1 if item["code"] >= 0 else 0)
    return Dataset.from_dict({"text": texts, "label": labels})

def code(model_id: str, data: List[Response], reporter: Callable[[str, bool], None]) -> List[Response]:
    tokenizer = BertTokenizer.from_pretrained(f"./data/{model_id}")
    model = BertForSequenceClassification.from_pretrained(f"./data/{model_id}")
    for datapoint in data:
        inputs = tokenizer(datapoint['value'], return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class = torch.argmax(logits).item()
        datapoint.status = 'CODE_SELECTION_PENDING'
        datapoint.codes = [predicted_class]
    return data

def train(
    task_label: str,
    instructions: TaskInstructions,
    input_data: List[Response],
    reporter: Callable[[str, bool], None]
) -> TrainingResult:
    coder_id = str(uuid.uuid4())
    dataset = to_dataset(input_data)
    finetune_model(dataset, coder_id, reporter)
    return TrainingResult(coderId = coder_id, msg = "okay")

def coder_exists(coder_id: str) -> bool:
    return redis_store.exists('instructions:' + coder_id)
