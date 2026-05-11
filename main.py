import uproot
import numpy as np
import pickle
from tqdm import tqdm

import os
import json
import glob

files = [
    "input/mva_ntuple_vbfy_preselection_030426CAFVBFHY_sig_X_c20a_vbfhyww_Herwig7.root", 
    "input/mva_ntuple_vbfy_preselection_030426CAFVBFHY_sig_X_c20d_vbfhyww_Herwig7.root",
    "input/mva_ntuple_vbfy_preselection_030426CAFVBFHY_sig_X_c20e_vbfhyww_Herwig7.root"
    ]
checkpoint_interval = 10000
output_dir = "output/"

with open("columns.json") as f:
    columns = json.load(f)

def find_negative(arr):
    # Find the indices of negative values
    negative_indices = np.where(arr < 0)[0]
    
    # If there are negative values, get the index of the first one
    if len(negative_indices) > 0:
        first_negative_index = negative_indices[0]
    else:
        first_negative_index = None  # No negative value found
    return first_negative_index

def least_negative_selection(data, target_column: str = "weight"):
    # Get the indices that would sort the 'weight' array
    sorted_indices = np.argsort(data[target_column])
    reversed = np.flip(sorted_indices)
    # Sort all arrays based on the sorted indices
    data = {key: value[reversed] for key, value in data.items()}
    ind = find_negative(data[target_column])
    return ind, data

def most_negative_selection(data, target_column: str = "weight"):
    # Get the indices that would sort the 'weight' array
    sorted_indices = np.argsort(data[target_column])
    # Sort all arrays based on the sorted indices
    data = {key: value[sorted_indices] for key, value in data.items()}
    ind = 0 if data[target_column][0] < 0 else None
    return ind, data

def inorder_selection(data, target_column: str = "weight"):
    ind = find_negative(data[target_column])
    return ind, data

def nearest_neighbors(data, ind, target_column, excluded_columns = ["j3", "btag", "m_dilepton_transverse"]):
    col_tmp = list(filter(lambda x: target_column != x and all(excluded not in x for excluded in excluded_columns),
                           columns))

    quadrutures = [(data[column] - data[column][ind])**2 for column in col_tmp]
    
    # Compute distances
    distances = np.sqrt(sum(quadrutures))
    
    # Find nearest neighbors excluding the index itself
    nearest_indices = np.argsort(distances)
    cumweight = np.cumsum(data[target_column][nearest_indices])
    positive_mask = cumweight > 0
    if np.any(positive_mask):
        nearest = np.argmax(positive_mask)
    else:
        nearest = len(nearest_indices) - 1
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

def normalize_arrays(data):
    """Convert uproot/pickle outputs to a plain dict of NumPy arrays."""
    if isinstance(data, np.ndarray) and data.dtype.names is not None:
        return {name: np.asarray(data[name]) for name in data.dtype.names}
    if isinstance(data, np.ndarray):
        raise TypeError(f"Unsupported ndarray shape for event data: {data.shape}")
    return data


def _branch_type(arr):
    """Infer a TTree branch type spec for uproot.mktree."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.dtype
    return np.dtype((arr.dtype, arr.shape[1:]))

def checkpoint(data, filename, output_dir, checkpoint_counter, checkpoint_interval):
    if checkpoint_counter % checkpoint_interval == 0:
        print(f"{checkpoint_counter} events resampled\nSaving file")
        base_name = filename.split('/')[-1]
        output_file = f"{output_dir}checkpoint_{checkpoint_counter}_{base_name}"
        
        try:
            arrays = {key: np.asarray(value) for key, value in data.items()}
            branch_types = {key: _branch_type(value) for key, value in arrays.items()}
            with uproot.recreate(output_file) as f:
                tree = f.mktree("HyWW", branch_types)
                tree.extend(arrays)
            print(f"Saved ROOT checkpoint to {output_file}")
        except Exception as e:
            print(f"CRITICAL: Failed to save ROOT checkpoint: {e}")
            return

        # Enforce output format: must be a TTree
        try:
            with uproot.open(output_file) as f:
                out_class = f["HyWW"].classname
            if out_class != "TTree":
                print(f"CRITICAL: Checkpoint class mismatch, expected TTree but got {out_class}")
            else:
                print("Checkpoint class verified: TTree")
        except Exception as e:
            print(f"WARNING: Could not verify checkpoint class: {e}")

def main(file_name, output_dir, target_column = "weight", is_pkl=False, tree_name="HyWW"):
    """
    Resamples one column from file with negative weights
    """
    if is_pkl:
        with open(file_name, 'rb') as file:
            data = pickle.load(file)
    else:
        with uproot.open(file_name) as file:
            data = file[tree_name].arrays(library="np")

    data = normalize_arrays(data)

    if isinstance(data, dict):
        first_key = next(iter(data))
        print(f"Loaded {len(data[first_key])} events from {file_name}")
        print(f"Data keys: {list(data.keys())[:5]}... ({len(data)} total)")
        print(data[target_column])

    # Main Loop for cell resampling
    print(f"Started cell resampling for {file_name}")
    checkpoint_counter = 0
    while True:
        # Recompute negatives from the current weights after every pass.
        negative_indices = np.where(data[target_column] < 0)[0]
        if len(negative_indices) == 0:
            break

        progress_made = False
        for ind in tqdm(negative_indices):
            if data[target_column][ind] >= 0:
                continue

            # ind, data = most_negative_selection(data)
            cell = nearest_neighbors(data, ind, target_column)
            data = resample(data, cell, target_column)

            # Increment checkpoint counter
            checkpoint_counter += 1
            progress_made = True
            checkpoint(data, file_name, output_dir, checkpoint_counter, checkpoint_interval)

        if not progress_made:
            print("WARNING: No progress made in this pass; stopping to avoid an infinite loop.")
            break
    
    if checkpoint_counter == 0:
        checkpoint(data, file_name, output_dir, checkpoint_counter, checkpoint_interval=checkpoint_interval)
    else:
        checkpoint(data, file_name, output_dir, checkpoint_counter, checkpoint_interval=checkpoint_counter)
    print(f"Done cell resampling for column {target_column}")

def iterate_over_columns(file_name, output_dir, target_columns = ["weight"]):
    base_name = file_name.split('/')[-1]
    checkpoint_pattern = f"{output_dir}checkpoint_*_{base_name}"
    stale_checkpoints = glob.glob(checkpoint_pattern)
    if stale_checkpoints:
        print(f"Removing {len(stale_checkpoints)} stale checkpoints for {base_name}")
        for cp in stale_checkpoints:
            os.remove(cp)

    for target_column in tqdm(target_columns):
        main(file_name, output_dir, target_column=target_column)


if __name__ == "__main__":
    file_name = files[0]
    for file_name in files:
        checkpoint_name = file_name.split("/")[-1].split(".")[-2]
        iterate_over_columns(file_name, output_dir, target_columns = ["weight"])
        print(f"Done cell resampling for file {checkpoint_name}")
