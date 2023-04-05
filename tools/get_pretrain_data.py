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
        a = i[1].split()
        b = i[2].split()
        samples.append(len(a)+len(b))

print(np.array(samples).max())
