import os
import pickle, zlib
import numpy as np
from decimal import *
from collections import defaultdict
import torch
import src.decompression.decompress as decompress
import src.compression.compress as compress

def compress_set(filename : str, saveloc : str):
    """
    @param filename : Filename of the set of MLP models to compress.
    @param saveloc : Save location for the MLP models.

    Writes compressed set into a seperate directory.
    """
    models = []
    for model in os.listdir(filename):
        if "._" in model:
            continue # Ignore hidden tar files
        models.append(model)
    compressed_deltas = {}
    base, bias = extract_weights(get_state_dict(filename + "/" + models[0]))
    compressed_deltas[models[0]] = (zlib.compress(base), bias)
    for m_ in models[1:]:
        print("Delta Compression on: {}".format(m_))
        curr, bias = extract_weights(get_state_dict(filename + "/" + m_))
        δt = np.subtract(curr, base)
        compressed_δt, lossy_δt = compress.compress_data(δt)
        compressed_deltas[m_] = (compressed_δt, bias)
        base = np.add(base, lossy_δt)
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)
    for key, val in compressed_deltas.items(): # Save process
        print("Saving Compressed Format: {}".format(key))
        idiv_file_path = os.path.join(saveloc, 'compressed_{}.pt'.format(key[:-4]))
        with open(idiv_file_path, 'wb') as f:
            pickle.dump(val, f)

def read_decompressed_state_dict(filepath):
    """
    @param filepath : Filepath where decompressed state_dict resides.

    @return The decompressed state_dict in dictionary format.
    """
    with open(filepath, 'rb') as f:
            return pickle.load(f)
        
def extract_weights(sd):
    """
    @param sd : Initial state_dict of the model.

    @return The base for all delta calculations.
    """
    weights = []
    bias = {}
    for layer_name, weight in sd.items():
        if 'bias' in layer_name:
            bias[layer_name] = weight
            continue
        weights.append(weight)
    return np.concatenate([tensor.flatten().numpy() for tensor in weights]), bias

def get_state_dict(filename : str) -> dict:
    """
    @param filename : Model checkpoint file.

    @return State dictionary of original model.
    """
    print("Loading: {}".format(filename))
    return torch.load(filename, map_location = torch.device('cpu'))["network_fn_state_dict"]


def load_compressed_set(filepath : str, saveloc : str, original_weight_dict : dict):
    """
    @param filepath : Filepath of set of MLP to load checkpoints from.
    @param saveloc : Filepath to save restored models to.
    @param original_weight_dict : The structure for the model to load into;
        Can be a randomly initialised weight dictionary.
    
    @return The decompressed MLP state dicts.
    """
    compressed_models = {}
    for model in os.listdir(filepath):
        with open(filepath + "/" + model, 'rb') as f:
            model_ = pickle.load(f)
        compressed_models[model] = model_
    base = None
    start = False
    for name, encoded_checkpoint in compressed_models.items():
        print("Decompressing for: {}".format(name))
        decoded_checkpoint = decompress.decode_data(encoded_checkpoint[0])
        bias = encoded_checkpoint[1]
        if not start: 
            base = decoded_checkpoint
            start = True
        else:
            base = np.add(base, decoded_checkpoint)
        base_dict = original_weight_dict.copy()
        compressed_models[name] = decompress.restore_state_dict(base, bias, base_dict)
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)
    for name, full_state_dict in compressed_models.items():
        new_name = "decompressed_" + name.split("compressed_")[-1]
        print("Saving Decompressed Model at: {}".format(new_name))
        idiv_file_path = os.path.join(saveloc, new_name)
        with open(idiv_file_path, 'wb') as f:
            pickle.dump(full_state_dict, f)