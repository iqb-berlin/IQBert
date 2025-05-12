from pathlib import Path

from ics import iqbert

model_path = Path("/app/data/bert-finetuned")
print(model_path.absolute())

text = "marvellous but disgusting"


iqbert.predict(model_path, text)