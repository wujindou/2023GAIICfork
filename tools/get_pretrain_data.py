import os
import csv
import numpy as np
from tokenizers import ByteLevelBPETokenizer
from transformers import BartTokenizer, BartForConditionalGeneration

samples = []
with open("../data/raw.csv",'r') as fp:
    reader = csv.reader(fp)
    sample = [row for row in reader]
    for i in sample:
        samples.append(i[1]+","+i[2])

# with open("../data/preliminary_a_test.csv",'r') as fp:
#     reader = csv.reader(fp)
#     sample = [row for row in reader]
#     for i in sample:
#         samples.append(i[1]+",")

with open("data.csv", 'w') as f:
    for i in samples:
        f.write(i + '\n')
