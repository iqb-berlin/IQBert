from transformers import BertTokenizer, BertForSequenceClassification
import torch
import sys

from read_data import get_data, store_data, load_data_file

tokenizer = BertTokenizer.from_pretrained("./bert-finetuned")
model = BertForSequenceClassification.from_pretrained("./bert-finetuned")

data = load_data_file(sys.argv[1])
result = []

for datapoint in data:
    print("Input: ", datapoint['value'])
    inputs = tokenizer(datapoint['value'], return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class = torch.argmax(logits).item()


    print("Predicted class: ", predicted_class)
    result.append({"value": datapoint['value'], "class": predicted_class})

store_data(result, sys.argv[2])
