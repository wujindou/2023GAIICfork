import os
import pandas as pd
import numpy as np
from tokenizers import ByteLevelBPETokenizer
from transformers import BartTokenizer, BartForConditionalGeneration

raw = pd.read_csv("../data/raw.csv", header=None)
pred_a = pd.read_csv("../data/preliminary_a_test.csv", header=None)
pred_b = pd.read_csv("../data/preliminary_b_test.csv", header=None)
data = raw.iloc[:,[1,2]]
pred_a = pd.DataFrame(pred_a.iloc[:,1])
pred_b = pd.DataFrame(pred_b.iloc[:,1])
data = pd.concat([data, pred_a, pred_b], axis=0)
data.to_csv('./data.csv', header=None)
