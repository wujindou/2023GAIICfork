import pandas as pd
import random
raw_data_path = "../data/data.txt"

with open(raw_data_path, 'r') as f:
    data = f.readlines()
random.shuffle(data)
train_data = data[:-500]
val_data = data[-500:]
with open("../data/pretrain.txt", 'w') as f:
    f.writelines(train_data)
with open("../data/preval.txt", 'w') as f:
    f.writelines(val_data)
