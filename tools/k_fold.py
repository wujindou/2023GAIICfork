import pandas as pd
from sklearn.model_selection import KFold

# Set the path to the raw CSV file
raw_data_path = "../data/raw.csv"

# Read the raw CSV file into a Pandas DataFrame
raw_data = pd.read_csv(raw_data_path)

# Split the raw data into input and output variables
# Initialize a 5-fold cross-validation object
kf = KFold(n_splits=5, shuffle=True, random_state=1024)

# Loop over each fold
for i, (train_index, val_index) in enumerate(kf.split(raw_data)):
    
    # Split the data into training and validation sets
    train_data = raw_data.iloc[train_index]
    val_data = raw_data.iloc[val_index]
    
    # Save the training and validation sets to CSV files
    train_data.to_csv("../data/train_{}.csv".format(i), index=False)
    val_data.to_csv("../data/val_{}.csv".format(i), index=False)
