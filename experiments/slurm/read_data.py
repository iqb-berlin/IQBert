import json
import sys
import random

from datasets import Dataset


def partition(data: list, n: float) -> tuple[list, list]:
    total_size = len(data)
    edge = int(total_size * n)
    random.shuffle(data)
    return data[0:edge], data[edge:len(data)]


def to_dataset(data) -> Dataset:
    texts = []
    labels = []
    for item in data:
        texts.append(item["value"])
        labels.append(1 if item["code"] >= 0 else 0)
    return Dataset.from_dict({"text": texts, "label": labels})

def get_data(file_path) -> tuple[list, list]:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read/parse JSON from {file_path}: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print("JSON root element is not an array. Exiting.")
        sys.exit(1)

    total_size = len(data)
    set1, set2 = partition(data, 2/3)

    print(f"Dataset of {total_size} split into {len(set1)} and {len(set2)} samples.")
    return set1, set2

def store_data(data, file_path):
    try:
        with open(file_path, 'w') as fp:
            json.dump(data, fp)
    except Exception as e:
        print(f"Failed to write JSON to {file_path}: {e}")
        sys.exit(1)
