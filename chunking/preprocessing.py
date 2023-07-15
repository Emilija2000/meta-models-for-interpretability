import functools
import jax
from jax import flatten_util
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Dict, Tuple, Callable
import chex
import numpy as np

################
# utility functions

def pad_to_chunk_size(arr: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    '''Add zeros to pad to chunk size'''
    pad_size = -len(arr) % chunk_size
    padded = jnp.pad(arr, (0, pad_size))
    return padded

def chunk_array(x: ArrayLike, chunk_size: int) -> jax.Array:
    """Split an array into chunks of size chunk_size. 
    If not divisible by chunk_size, pad with zeros."""
    x = x.flatten()
    x = pad_to_chunk_size(x, chunk_size)
    return x.reshape(-1, chunk_size)

def skip_layer(layer_name: str) -> bool:
    """Skip certain layers when chunking and unchunking."""
    skip_list = ['dropout', 'norm']
    return any([s in layer_name.lower() for s in skip_list])

def filter_layers(
        params: Dict[str, Dict[str, ArrayLike]]
        ) -> Tuple[Dict[str, Dict[str, ArrayLike]], Callable]:
    """
    Filters out layers from params and provides a callable to retrieve them.
    """
    if not params:
        raise ValueError("Empty parameter dict.")

    output_layers = {}
    removed_layers = {}
    approved_layers = ['conv', 'linear', 'head', 'mlp']
    for k, v in params.items():
        if skip_layer(k):
            removed_layers[k] = v
        elif any([l in k.lower() for l in approved_layers]):
            output_layers[k] = v
        else:
            raise ValueError(f"Invalid layer: {k}.")

    original_order = list(params.keys())
    
    def unfilter(filtered_params) -> Dict[str, Dict[str, ArrayLike]]:
        """Inverse of filter_layers."""
        return {k: (filtered_params[k] if k in filtered_params
                     else removed_layers[k]) for k in original_order}
        
    return output_layers, unfilter

def get_param_shapes(
        params: Dict[str, Dict[str, ArrayLike]]) -> Dict[str, Dict[str, Tuple[int, ...]]]:
    return {
        k: {subk: v[0].shape for subk, v in layer.items()}
        for k, layer in params.items()
    }

########################
# flat preprocessing

def preprocess(
        params: Dict[str, Dict[str, ArrayLike]], 
        chunk_size: int
        ) -> Tuple[jax.Array, Callable]:
    """Preprocess a pytree of parameters into a flat array of chunks."""
    params, unfilter = filter_layers(params)
    flat_params, unflatten = flatten_util.ravel_pytree(params)
    padded = pad_to_chunk_size(flat_params, chunk_size)
    chunks = padded.reshape(-1, chunk_size)
    
    def unprocess(chunks: ArrayLike) -> Dict[str, Dict[str, ArrayLike]]:
        """Inverse of preprocess."""
        flat_params_new = chunks.flatten()[:len(flat_params)]
        return unfilter(unflatten(flat_params_new))

    return chunks, unprocess

def get_unpreprocess(
        params: Dict[str, Dict[str, ArrayLike]],
        chunk_size: int,
        verbose: bool = True,
        ) -> Tuple[Dict[str, Dict[str, Tuple[int, ...]]], int]:
    """Preprocess once to get the unpreprocess function."""
    params, _ = filter_layers(params)
    chunks, unpreprocess = preprocess(params, chunk_size)
    raveled_params = flatten_util.ravel_pytree(params)[0]
    if verbose:
        print()
        print(f"Number of (relevant) layers per net: {len(params)}")
        print(f"Number of parameters per net: "
            f"{raveled_params.shape[0]}")
        print(f"Chunk size: {chunk_size}")
        print(f"Number of chunks per net: {chunks.shape[0]}")
        print()
    return unpreprocess

#######################
# chunk by layer
def preprocess_layerwise(
        params: Dict[str, Dict[str, ArrayLike]], 
        chunk_size: int
        ) -> Tuple[jax.Array, Callable]:
    """Preprocess a pytree of parameters into a flat array of chunks,
    where each layer is chunked separately."""
    params, unfilter = filter_layers(params)
    
    # Get original shapes
    param_shapes = jax.tree_util.tree_map(lambda x: x.shape, params)
    flat_params = jax.tree_util.tree_map(lambda x: x.flatten(), params)
    original_lengths = jax.tree_util.tree_map(lambda x: len(x), flat_params)
    flat_params = jax.tree_util.tree_map(lambda x: chunk_array(x, chunk_size), flat_params)
    
    # flatten to array of chunks
    flat_params, unflatten = flatten_util.ravel_pytree(flat_params)
    chunked = flat_params.reshape(-1, chunk_size)
    
    def unprocess(chunks: ArrayLike) -> Dict[str, Dict[str, ArrayLike]]:
        """Inverse of preprocess."""
        flat_params_new = chunks.flatten()[:len(flat_params)]
        params = unflatten(flat_params_new)
        params = jax.tree_util.tree_map(lambda x, orig_len: x.flatten()[:orig_len], params, original_lengths)
        params = jax.tree_util.tree_map(lambda x, shape: x.reshape(shape), params, param_shapes)
        return unfilter(params)
    
    return chunked, unprocess

def get_unpreprocess_layerwise(
        params: Dict[str, Dict[str, ArrayLike]],
        chunk_size: int,
        verbose: bool = True,
        ) -> Callable:
    """Preprocess once to get the unpreprocess function."""
    params, _ = filter_layers(params)
    chunks, unpreprocess = preprocess_layerwise(params, chunk_size)
    param_shapes = get_param_shapes(params)
    raveled_params = flatten_util.ravel_pytree(params)[0]
    if verbose:
        print()
        print(f"Number of (relevant) layers per net: {len(params)}")
        print(f"Number of parameters per net: "
            f"{raveled_params.shape[0]}")
        print(f"Chunk size: {chunk_size}")
        print(f"Number of chunks per net: {chunks.shape[0]}")
        print()
    return unpreprocess

########################
# Added indicators
def preprocess_withindicators(
        params: Dict[str, Dict[str, ArrayLike]], 
        chunk_size: int
        ) -> Tuple[jax.Array, Callable]:
    """Preprocess a pytree of parameters into a flat array of chunks,
    where each layer is chunked separately, with indicators for layer type
    and with indicators for end-of-layer."""
    
    # Define the layer types
    layer_types = {"conv": 0, "linear": 1, "other": 2}
    num_layer_types = len(layer_types)
    def get_layer_type(layer) -> int:
        for type_name in layer_types:
            if type_name in layer.key.lower():
                return layer_types[type_name]
        return layer_types["other"]
    
    # filter out some layer types
    params, unfilter = filter_layers(params)
    
    # Get original shapes and flatten parameters
    param_shapes = jax.tree_util.tree_map(lambda x: x.shape, params)
    flat_params = jax.tree_util.tree_map(lambda x: x.flatten(), params)
    original_lengths = jax.tree_util.tree_map(lambda x: len(x), flat_params)

    # Chunk params
    chunked_params = jax.tree_util.tree_map(lambda x: chunk_array(x, chunk_size), flat_params)
    
    def add_embeddings(path, chunk):
        layer = path[0]
        layer_type_idx = get_layer_type(layer)
        layer_type_one_hot = [0] * num_layer_types
        layer_type_one_hot[layer_type_idx] = 1

        # Prepare embeddings
        embeddings = jnp.tile(jnp.array(layer_type_one_hot), (len(chunk), 1))

        # Prepare end of layer token
        end_token = np.zeros((len(chunk), 1))
        if len(chunk) > 0:
            end_token[-1, 0] = 1
        end_token = jnp.array(end_token)

        # Combine chunks, embeddings, and end token
        combined = jnp.hstack([chunk, embeddings, end_token])

        return combined
    
    embedded_params = jax.tree_util.tree_map_with_path(add_embeddings, chunked_params)

    # Flatten to array of chunks
    flat_params, unflatten = flatten_util.ravel_pytree(embedded_params)
    chunked = flat_params.reshape(-1, chunk_size + num_layer_types + 1)  
    
    def unprocess(chunks: ArrayLike) -> Dict[str, Dict[str, ArrayLike]]:
        """Inverse of preprocess."""
        orig_len = chunks.shape[1] - num_layer_types - 1
        chunks = chunks[:, :orig_len]
        params = unflatten(chunks.flatten()[:len(flat_params)])
        params = jax.tree_util.tree_map(lambda x, orig_len: x.flatten()[:orig_len], params, original_lengths)
        params = jax.tree_util.tree_map(lambda x, shape: x.reshape(shape), params, param_shapes)
        return unfilter(params)
    
    return chunked, unprocess

def get_unpreprocess_withindicators(
        params: Dict[str, Dict[str, ArrayLike]],
        chunk_size: int,
        verbose: bool = True,
        ) -> Callable:
    """Preprocess once to get the unpreprocess function."""
    params, _ = filter_layers(params)
    chunks, unpreprocess = preprocess_withindicators(params, chunk_size)
    param_shapes = get_param_shapes(params)
    raveled_params = flatten_util.ravel_pytree(params)[0]
    if verbose:
        print()
        print(f"Number of (relevant) layers per net: {len(params)}")
        print(f"Number of parameters per net: "
            f"{raveled_params.shape[0]}")
        print(f"Chunk size: {chunk_size}")
        print(f"Number of chunks per net: {chunks.shape[0]}")
        print()
    return unpreprocess

#######################
# Check for high variance or mean of params

def flatten(x):
    return flatten_util.ravel_pytree(x)[0]

def is_fine(params: dict):
    """Return false if std or mean is too high."""
    flat = flatten(params)
    if flat.std() > 5.0 or jnp.abs(flat.mean()) > 5.0:
        return False
    else:
        return True

# TODO untested!
def filter_data(*arrays):
    """Given a list of net arrays, filter out those
    with very large means or stds."""
    chex.assert_equal_shape_prefix(arrays, 1)  # equal len

    def all_fine(elements):
        return all([is_fine(x) for x in elements])

    arrays_filtered = zip(*[x for x in zip(*arrays) if all_fine(x)])
    num_filtered = len(arrays[0]) - len(arrays_filtered[0])
    print(f"Filtered out {num_filtered} nets.")
    return arrays_filtered

##################
# Fast versions without unpreprocess

@functools.partial(jax.jit,static_argnums=(1))
def fast_process_layerwise(params: Dict[str, Dict[str, ArrayLike]], chunk_size: int) -> jax.Array:
    flat_params = jax.tree_util.tree_map(lambda x: x.flatten(), params)
    flat_params = jax.tree_util.tree_map(lambda x: chunk_array(x, chunk_size), flat_params)
    flat_params, _ = flatten_util.ravel_pytree(flat_params)
    chunked = flat_params.reshape(-1, chunk_size)
    return chunked

def batch_process_layerwise(param_list, chunk_size):
    return jnp.stack(list(map(functools.partial(fast_process_layerwise, chunk_size=chunk_size), param_list)))
