import sys
from data import to_dataset, get_data, store_data
from iqbert import finetune_model

set1, set2 = get_data(sys.argv[1])

print('input file = ' + sys.argv[1])
print('output file 1 = ' + sys.argv[2])
print('output file 2 = ' + sys.argv[3])

store_data(set1, sys.argv[2])
store_data(set2, sys.argv[3])

