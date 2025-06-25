import sys
from read_data import to_dataset, get_data, store_data
from iqbert import finetune_model

set1, set2 = get_data(sys.argv[1])

print('input file = ' + sys.argv[1])
print('output file = ' + sys.argv[2])

store_data(set2, sys.argv[2])
d1 = to_dataset(set1)
finetune_model(d1)

