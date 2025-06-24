import sys

from read_data import get_data, store_data
from iqbert import finetune_model

set1, set2 = get_data(sys.argv[1])
finetune_model(set1)
store_data(set2, sys.argv[2])
