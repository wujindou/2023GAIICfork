import sys
sys.path.append('..')
import torch
from models_bart import CustomBartModel

num_folds = 5

# Define the list of checkpoint paths
checkpoint_paths = ['./checkpoint/1_{}/model_cider.pt'.format(i) for i in range(num_folds)]

# Initialize the model with the configuration
model = CustomBartModel()

# Define the dictionary to store the averaged weights
average_weights = {}

# Loop over the checkpoints and accumulate the weights
for checkpoint_path in checkpoint_paths:
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Get the model state_dict from the checkpoint
    state_dict = checkpoint['model']
    
    # Loop over the state_dict and accumulate the weights
    for key, value in state_dict.items():
        # key = key[6:]
        if key not in average_weights:
            average_weights[key] = value.clone().detach()
        else:
            average_weights[key] = value.clone().detach()

# Compute the average weights
for key in average_weights:
    average_weights[key] /= num_folds

# Load the averaged weights into the model
model.load_state_dict(average_weights)
# Save the averaged weights to a file
torch.save(average_weights, 'averaged_model_weights.pt')
averaged_weights = torch.load('averaged_model_weights.pt')
model = CustomBartModel()
model.load_state_dict(averaged_weights)
