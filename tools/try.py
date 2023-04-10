import pandas as pd
data = pd.read_csv("../data/preliminary_a_test.csv", header=None)
for i in range(3000):
    print(data.iloc[i,1]+",")
