from chunking import preprocessing

import jax
import jax.numpy as jnp


def mask_data(rng, inputs, mask_token=0., mask_prob:float=0.15, individual_w:bool=False, binary_indicator:bool=True, resample_zeromasks:bool=True):
    '''
    Masks input sequence
    
    Args:
        rng (jax.random.PRNGKey) - random seed used to sample chunks or weights to be masked
        inputs (list) - sequence of input weights to be masked
        mask_token (float or ArrayLike) - token that will replace masked weights, should be float if 
            individual_w==True, and array of chunk shape otherwise or something that could be casted 
            into an array of correct dimensions)
        mask_prob (float) - masking probability
        individual_w (bool) - if true, each weight is masked individually, otherwise mask whole chunks 
        binary_indicator (bool) - if true, additional binary values are added (concatinated) to the 
            network input to indicate which weights are masked
            
    Result:
        maksed input data 
        mask - binary matrix showing which outputs should be used in loss calculation 
    '''
    masked_inputs = []
    masked_positions = []
    non_masked_net_positions=[]

    for seq in inputs:
        # randomly choose weights to mask
        mask_nonzero_flag = False
        while not mask_nonzero_flag:
            rng, subkey = jax.random.split(rng)
            if individual_w:
                mask = jax.random.uniform(subkey,seq.shape) < mask_prob
            else:
                mask = jax.random.uniform(subkey,(seq.shape[0],1)) < mask_prob
            
            mask_nonzero_flag = jnp.sum(mask) > 0
            if not(resample_zeromasks):
                mask_nonzero_flag = True
                
        # replace weights with mask token
        masked_seq = jnp.copy(seq)
        masked_seq = masked_seq.at[jnp.where(mask)[0]].set(mask_token)
        
        # optionally add binary mask indicators - 1 means masked (important)!
        if binary_indicator:
            masked_ind = 1.0*mask
            masked_seq = jnp.concatenate([masked_seq, masked_ind], axis=1)
         
        non_masked_net = 1 - mask
        # do not consider indicator part output in loss calc -> add zeros to mask
        # TODO: should they be considered??
        original_mask_shape = mask.shape
        if not(individual_w):
            mask = jnp.tile(mask, (1,seq.shape[1]))
            non_masked_net = jnp.tile(non_masked_net, (1,seq.shape[1]))
        if binary_indicator:
            mask = jnp.concatenate([mask, jnp.zeros(original_mask_shape)],axis=1)
            non_masked_net = jnp.concatenate([non_masked_net, jnp.zeros(original_mask_shape)],axis=1)
            
        masked_inputs.append(masked_seq) 
        masked_positions.append(jnp.asarray(mask, dtype=jnp.int32))
        non_masked_net_positions.append(jnp.asarray(non_masked_net, dtype=jnp.int32))

    return masked_inputs, masked_positions, non_masked_net_positions

def process_batch(rng, inputs, mask_token=None, mask_prob=0, chunk_size=100, mask_individual=False, mask_indicators=True,resample_zeromasks=True, layerwise=True):
    '''Output masked inputs, "labels" and binary matrix of masked positions'''
    # chunk weights (tokenize) and mask
    if layerwise:
        inputs = [preprocessing.preprocess_layerwise(inp, chunk_size)[0] for inp in inputs]
    else: 
        inputs = [preprocessing.preprocess(inp, chunk_size)[0] for inp in inputs]
    
    masked_inputs, masked_positions, non_masked_positions = mask_data(rng, inputs, mask_token, mask_prob,individual_w=mask_individual, binary_indicator=mask_indicators, resample_zeromasks=resample_zeromasks)
    # pad labels to the correct shape (if indicator tokens are added to masked_inputs)
    labels = []
    for inp, m_inp in zip(inputs, masked_inputs):
        p = m_inp.shape[1] - inp.shape[1]
        labels.append(jnp.pad(inp, pad_width=((0,0), (0, p)), constant_values=0))
        #TODO: add different labels padding if you want to use both parts in loss calculation
    # for batches of different models - pad all sequences to the biggest model size
    masked_inputs = pad_and_stack_arrays(masked_inputs)
    labels = pad_and_stack_arrays(labels)
    masked_positions = pad_and_stack_arrays(masked_positions)
    masked_positions = masked_positions.astype(bool)
    non_masked_positions = pad_and_stack_arrays(non_masked_positions)
    
    return masked_inputs, labels, masked_positions, non_masked_positions

def pad_and_stack_arrays(arrays):
    '''given a list of arrays of shape (x_i,y) for i in {1..arrays len}, pad along dim 0 and stack'''
    x_max = max(arr.shape[0] for arr in arrays)
    padded_arrays = []
    for arr in arrays:
        if arr.shape[0] < x_max:
            pad_width = ((0, x_max - arr.shape[0]), (0, 0))  # Pad only along the first dimension
            padded_arr = jnp.pad(arr, pad_width, mode='constant')
        else:
            padded_arr = arr
        padded_arrays.append(padded_arr)
    stacked_array = jnp.stack(padded_arrays, axis=0)
    return stacked_array