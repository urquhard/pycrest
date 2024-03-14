import uproot
import numpy as np
import pickle
from tqdm import tqdm

import os

files = ["/home/nda70/projects/ctb-stelzer/nda70/nTuples-cHj3-cHW-12M.root", "/home/nda70/projects/ctb-stelzer/nda70/top.root"]
checkpoint_interval = 1000
output_dir = "/home/nda70/projects/ctb-stelzer/nda70/"

# Target columns excluding weight_sm and weight
with open("columns.txt") as f:
    d = f.read()
l = d.split('\t')
target_columns = list(filter(lambda x: "weight" in x and not "int" in x and not "bsm" in x, l))[1:-1]

def find_negative(arr):
    # Find the indices of negative values
    negative_indices = np.where(arr < 0)[0]
    
    # If there are negative values, get the index of the first one
    if len(negative_indices) > 0:
        first_negative_index = negative_indices[0]
    else:
        first_negative_index = None  # No negative value found
    return first_negative_index

def least_negative_selection(data):
    # Get the indices that would sort the 'weight_sm' array
    sorted_indices = np.argsort(data['weight_sm'])
    reversed = np.flip(sorted_indices)
    # Sort all arrays based on the sorted indices
    data = {key: value[reversed] for key, value in data.items()}
    ind = find_negative(data['weight_sm'])
    return ind, data

def most_negative_selection(data):
    # Get the indices that would sort the 'weight_sm' array
    sorted_indices = np.argsort(data['weight_sm'])
    # Sort all arrays based on the sorted indices
    data = {key: value[sorted_indices] for key, value in data.items()}
    ind = 0 if data['weight_sm'][0] < 0 else None
    return ind, data

def inorder_selection(data):
    ind = find_negative(data['weight_sm'])
    return ind, data

def nearest_neighbors(data, ind, target_column):
    lep0_pt = data['lep0_pt']
    lep1_pt = data['lep1_pt']
    jet0_pt = data['jet0_pt']
    jet1_pt = data['jet1_pt']
    
    # Compute distances
    distances = np.sqrt((lep0_pt - lep0_pt[ind])**2 + (lep1_pt - lep1_pt[ind])**2 + (jet0_pt - jet0_pt[ind])**2 + (jet1_pt - jet1_pt[ind])**2)
    
    # Find nearest neighbors excluding the index itself
    nearest_indices = np.argsort(distances)
    cumweight = np.cumsum(data[target_column][nearest_indices])
    nearest = np.argmax(cumweight>0)
    # nearest = np.searchsorted(aa, N/2)+1 # Slightly faster
    cell = nearest_indices[0:nearest + 1]
    return cell

def resample(data, nearest_indices, target_column):
    # Extract relevant arrays
    sm_weights = data[target_column]
    
    # Compute sum of absolute values and sum of values
    abs_sum = np.sum(np.abs(sm_weights[nearest_indices]))
    sum_values = np.sum(sm_weights[nearest_indices])
    
    # Update values according to the formula
    sm_weights[nearest_indices] = sum_values / abs_sum * np.abs(sm_weights[nearest_indices])
    data[target_column] = sm_weights
    return data

def checkpoint(data, filename, output_dir, checkpoint_counter, checkpoint_interval):
    filename = filename.split("/")[-1].split(".")[-1]
    # Check if it's time for a checkpoint
    if checkpoint_counter % checkpoint_interval == 0:
        # Save checkpoint to pickle file
        print(f"{checkpoint_counter} events resampled\nSaving file")
        checkpoint_filename = f"{output_dir}{filename}.pkl"
        with open(checkpoint_filename, 'wb') as f:
            pickle.dump(data, f)

def main(file_name, output_dir, target_column = "weight_sm", is_pkl=False):
    """
    Resamples one column from file with negative weights
    """
    if is_pkl:
        with open(file_name, 'rb') as file:
            data = pickle.load(file)
    else:
        with uproot.open(file_name) as file:
            data = file["HWWTree_emme"].arrays(library="np")

    # Main Loop for cell resampling
    print(f"Started cell resampling for {file_name}")
    checkpoint_counter = 0
    # Find the indices of negative values
    negative_indices = np.where(data[target_column] < 0)[0]
    used = np.zeros_like(negative_indices)
    for ind in tqdm(negative_indices):
        if (ind in negative_indices * used and ind != 0):
            continue
        # ind, data = most_negative_selection(data)
        cell = nearest_neighbors(data, ind, target_column)
        data = resample(data, cell, target_column)
        
        # Remove the resampled indices from negative_indices
        indices_in_negative = np.where(np.isin(negative_indices, cell))[0]
        used[indices_in_negative] = 1

        # Increment checkpoint counter
        checkpoint_counter += 1
        checkpoint(data, file_name, output_dir, checkpoint_counter, checkpoint_interval)
    
    checkpoint(data, file_name, output_dir, checkpoint_counter, checkpoint_interval=checkpoint_counter)
    print(f"Done cell resampling for column {target_column}")


if __name__ == "__main__":
    file_name = files[0]
    checkpoint_name = file_name.split("/")[-1].split(".")[-2]
    checkpoint_filename = f"{output_dir}{checkpoint_name}.pkl"
    if os.path.exists(checkpoint_filename):
        main(checkpoint_filename, output_dir, target_column = target_columns[0], is_pkl=True)
    else:
        main(file_name, output_dir)
    print(f"Done cell resampling for file {checkpoint_name}")
