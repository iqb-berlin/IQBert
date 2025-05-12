from ics import iqbert

data = iqbert.load_data()
print(data['train'].data[0][1000])
print(data['train'].data[1][1000])
# iqbert.finetune_model(data)

