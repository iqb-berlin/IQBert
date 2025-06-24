import sys
from experiments.slurm.read_data import to_dataset
from read_data import get_data, store_data
from iqbert import finetune_model

set1, set2 = get_data(sys.argv[1])

d1 = to_dataset(set1)
print(d1)
finetune_model(d1)
store_data(set2, sys.argv[2])
