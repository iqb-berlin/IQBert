from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer.name_or_path)