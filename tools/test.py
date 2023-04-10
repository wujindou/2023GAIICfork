import pandas as pd
import numpy as np
data = pd.read_csv("../data/data.csv", header=None)
for i in range(23000):
    a,b = data.iloc[i]
    print(pd.isna(b))
