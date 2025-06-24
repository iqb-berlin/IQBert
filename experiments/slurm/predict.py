from transformers import BertTokenizer, BertForSequenceClassification
import torch
import sys

if len(sys.argv) < 2:
    print('no argument')
    sys.exit()

# Load from the saved directory
tokenizer = BertTokenizer.from_pretrained("./bert-finetuned")
model = BertForSequenceClassification.from_pretrained("./bert-finetuned")


for i in range(1, len(sys.argv)):
    # Prepare input
    text = sys.argv[i]
    inputs = tokenizer(text, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class = torch.argmax(logits).item()

    print("Input: ", sys.argv[i])
    print("Predicted class: ", predicted_class)
