import json
import sys
import random

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)], [list_in[i2::n] for i2 in range(n, len(list_in))]


file_path = sys.argv[1]
try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except Exception as e:
    print(f"Failed to read/parse JSON: {e}")
    sys.exit(1)

if not isinstance(data, list):
    print("JSON root element is not an array. Exiting.")
    sys.exit(1)

total_size = len(data)
split_1_size = int(total_size * 2 / 3)
split_2_size = total_size - split_1_size

set1, set2 = partition(data, split_1_size)

print(f"Dataset of {total_size} split into {len(set1)} and {len(set2)} samples.")
