import sys
from data import to_dataset, load_data_file
from iqbert import finetune_model

set = load_data_file(sys.argv[1])

print('input file = ' + sys.argv[1])

d = to_dataset(set)
finetune_model(d)

print('done')