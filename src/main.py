import os
import pickle
import numpy as np
import pathos.multiprocessing as pmp
from decimal import *
from collections import defaultdict
import dahuffman
import torch
import sys, math

__all__ = ['compress_data', 'decode_data', 'restore_model', 'restore_model_async',
           'save_checkpoint', 'load_checkpoint', 'calculate_compression_rate']

def extract_weights(initmodel):
    """
    @param initmodel : Initial run of the model.

    @return The base for all delta calculations.
    """
    wd = initmodel.state_dict()
    weights = []
    for k, v in wd.items():
        if "classifier" in k or "weight" not in k:
            continue
        weights.append(v)
    return np.concatenate([tensor.flatten().numpy() for tensor in weights])

def generate_delta(weights_prev : np.array, sd_curr):
    """
    @param sd_prev : The previous weights
    @param sd_curr : The state dictionary of model 2.

    @return The delta for the weights together with the bias and dense layers.
    Also returns the full current weight for benchmarking purpose.
    """
    weights_curr = []
    full = {} # Store layers that require full save
    for k in sd_curr:
        if "classifier" in k or "weight" not in k: # If not weight or conv layer: Store full config.
            full[k] = sd_curr[k]
            continue
        # Extract weights for prev and current layer.
        weights_curr.append(sd_curr[k])
        
    # Generate weight delta.
    curr_flatten = np.concatenate([tensor.numpy().flatten() for tensor in weights_curr])
    weight_delta = np.subtract(curr_flatten, weights_prev)
    
    return weight_delta, full, weights_curr

def compress_data(δt, num_bits = 3, threshhold=True):
    """
    @param δt : The delta to compress.
    @param num_bits : The number of bits to limit huffman encoded variables to.
    @param treshold : Enabler for priority promotion process.

    @return The huffman encoded delta, the huffman encoder as well.
    Lastly, returns the new delta to be used to update the current prev_state.
    """
    # Original delta is in base 10 representation, extract mantissa & sign from base 2 representation.
    _, δt_exp = np.frexp(δt) # Mantissa & exponent
    δt_sign = np.sign(δt) # Sign
    δt_sign[δt_sign > 0] = 0
    δt_sign[δt_sign < 0] = 1    
    
    # Sort elements by sign & exponent.
    mp =  defaultdict(list)
    for i in range(len(δt)):
        mp[(δt_exp[i], δt_sign[i])].append((i, δt[i]))

    # Within each bucket, take the average.
    for k in mp:
        mp[k] = (np.average(np.array([x[-1] for x in mp[k]])), 
                 [x[0] for x in mp[k]])
    mp = list(mp.values())

    # Priority Promotion.
    if threshhold:
        allowed_buckets = int(math.pow(2, num_bits) - 1)
        mp = sorted(mp, key = lambda x : abs(x[0]), reverse = True)[:min(allowed_buckets, len(mp))]

    # Restore original array after promotion
    new_δt= [0 for x in range(len(δt))]
    for qtVal, pos in mp:
        for p in pos:
            new_δt[p] = qtVal
    new_δt = np.array(new_δt)

    # Encode new_δt using Huffman coding
    encoder = dahuffman.HuffmanCodec.from_data(new_δt)
    encoded = encoder.encode(new_δt)
    return encoded, encoder, new_δt

def decode_data(encoded, encoder):
    """
    @param encoded : Encoded checkpoint
    @param encoder : Encoder used to encode checkpoint.

    @return : Decoded checkpoint.
    """
    return np.array(encoder.decode(encoded))

def save_checkpoint(filepath, compressed_data, full, epoch, iteration, encoder):
    """
    @param filepath : Filepath (.pt) to be stored in.
    @param compressed_data : Compressed data
    @param full : The dictionary containing the full layers for the model.
    @param epoch : Current epoch
    @param iteration : Current iteration
    @param encoder : Huffman Encoder associated to compressed checkpoint.
    """
    # Save the compressed data and epoch number to a file
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    fn = os.path.join(filepath, 'lc_checkpoint_epoch{}_iter{}.pt'.format(epoch, iteration))
    with open(fn, 'wb') as f:
        pickle.dump((compressed_data, full, encoder, epoch), f)

def load_checkpoint(filepath):
    """
    @param filepath : Filepath for checkpoint to be loaded from.
    
    @return The decompressed data, the bias dictionary and its associated epoch.
    """
    # Load the compressed data and epoch number from a file
    with open(filepath, 'rb') as f:
        compressed_data, full, encoder, epoch = pickle.load(f)
    decompressed_data = decode_data(compressed_data, encoder)  # decode the binary data
    return decompressed_data, full, epoch

def restore_model(init_model, last_epoch, last_iter, init_filepath, cpt_filename):
    """
    @param init_model : Clean version of original model.
    @param last_epoch : The last epoch of checkpoint generated by the model
    @param last_iter : The last iteration of checkpoint generated by the model.
    @param max_iter : The maximum number of iteration per epoch.
    @param init_filepath : The path leading to the 'base' copy of model weights.
    @param cpt_filename : The file containing all the checkpoints for given branch.

    @return Restored model.
    """
    # Get initial model
    init_model.load_state_dict(torch.load(init_filepath))
    temp  = init_model.state_dict()
    weights = [temp[x] for x in temp if "weight" in x and "classifier" not in x]
    deltas = np.concatenate([tensor.numpy().flatten() for tensor in weights])
    
    fin_dict = init_model.state_dict()

    # Get the latest bias & dense layer.
    ckpt = cpt_filename + "lc_checkpoint_epoch{}_iter{}.pt".format(last_epoch, last_iter)
    _, full, _ = load_checkpoint(ckpt)

    # Restore the bias & non-conv layers
    for f_layer, state in full.items():
        fin_dict[f_layer] = state

    # Get all checkpoints
    s = os.listdir(cpt_filename)
    filenames = [cpt_filename + x for x in s]
    
    for f in filenames:
        d, _, _ = load_checkpoint(f)
        deltas = np.add(d, deltas)

    # Reshape based on model's state dictionary.
    last_idx = 0
    
    for layer_name, init_tensor in fin_dict.items():
        # Ensure that only weights are being restored.
        if "classifier" in layer_name or "weight" not in layer_name:
            continue
        # Extract appropriate elements
        dim = init_tensor.numpy().shape
        if dim == ():
            continue
        t_elements = np.prod(dim)
        needed_ele = deltas[last_idx : last_idx + t_elements]
        last_idx += t_elements # Reset the last index

        # Reshape delta and reinsert into the dictionary.
        fin_dict[layer_name] = torch.unflatten(torch.from_numpy(needed_ele), -1, dim)
    
    init_model.load_state_dict(fin_dict)
    return init_model

def restore_model_async(init_model, last_epoch, last_iter,
                        init_filename, cpt_filename, 
                        n_cores = pmp.cpu_count() // 2):
    """
    @param init_model : Clean version of original model.
    @param last_epoch : The last epoch of checkpoint generated by the model
    @param last_iter : The last iteration of checkpoint generated by the model.
    @param max_iter : The maximum number of iteration per epoch.
    @param init_filepath : The path leading to the 'base' copy of model weights.
    @param cpt_filename : The file containing all the checkpoints for given branch.
    @param n_cores : The number of cores used to regenerate original model.

    @return Restored model.
    """
    # Get initial model
    init_model.load_state_dict(torch.load(init_filename))
    temp  = init_model.state_dict()
    weights = [temp[x] for x in temp if "weight" in x and "classifier" not in x]
    deltas = np.concatenate([tensor.numpy().flatten() for tensor in weights])
    print("Restored base model")

    print("Decompression process with {} cores".format(n_cores))

    pool = pmp.ProcessingPool(nodes = n_cores)
    filenames = []

    # Get all filenames within the sectional directory.
    s = os.listdir(cpt_filename)
    filenames = [cpt_filename + x for x in s]

    # Async checkpoint decompression
    with pool:
        results = pool.amap(lambda x : load_checkpoint(x), 
                             filenames)
        for res in results.get():
            d, _, _ = res
            deltas = np.add(deltas, d)

    # Reshape based on model's state dictionary.
    fin_dict = init_model.state_dict()
    last_idx = 0

    # Get the latest bias.
    ckpt = cpt_filename + "lc_checkpoint_epoch{}_iter{}.pt".format(last_epoch, last_iter)
    _, full, _ = load_checkpoint(ckpt)

    # Restore the bias and dense layer
    for f_layer, state in full.items():
        fin_dict[f_layer] = state

    # Restore the weights
    for layer_name, init_tensor in fin_dict.items():
        if "classifier" in layer_name or "weight" not in layer_name:
            continue
        # Extract appropriate elements
        dim = init_tensor.numpy().shape
        if dim == ():
            continue
        t_elements = np.prod(dim)
        needed_ele = deltas[last_idx : last_idx + t_elements]
        last_idx += t_elements # Reset the last index

        # Reshape delta and reinsert into the dictionary.
        fin_dict[layer_name] = torch.unflatten(torch.from_numpy(needed_ele), -1, dim)
    
    init_model.load_state_dict(fin_dict)
    return init_model