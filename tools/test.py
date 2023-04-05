import pandas as pd
data = pd.read_csv("./data.csv", header=None)
for i in range(22999):
    a,b = data.iloc[i]
    print(a,b)
