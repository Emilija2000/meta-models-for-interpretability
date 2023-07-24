'''
Loading model zoos published in: https://arxiv.org/pdf/2209.14764.pdf
Credit for zoo definition and loading: https://github.com/ModelZoos/ModelZooDataset/tree/main
'''
import jax.numpy as jnp
import numpy as np
import torch
from pathlib import Path
from torchload.checkpoints_to_datasets.dataset_base import ModelDatasetBase

def load_dataset_from_path(path:Path,epoch_lst:list=[5,15,25,50]):
    """
    Loads custom dataset class from raw zoo.
    input path: pathlib.Path to raw model zoo.
    input epoch_lst: list of integers, indicating the epochs of which to load the models 
    return dataset: dict with "trainset", "valset", "testset" 
    """
    # compose properties to map for
    result_key_list = [
        "test_acc",
        "training_iteration",
        "ggap",
    ]
    config_key_list = [
        "model::nlin", 
        "model::init_type",
        "optim::optimizer",
        "model::dropout",
        "optim::lr",
        "optim::wd"
    ]
    property_keys = {
        "result_keys": result_key_list,
        "config_keys": config_key_list,
    }

    layer_lst = [
        (0, "conv2d"),
        (3, "conv2d"),
        (6, "conv2d"),
        (9, "fc"),
        (11, "fc"),
    ]
    
    # set dataset path
    path_zoo_root = [path.absolute()]
        
    # load datasets
    # trainset
    trainset = ModelDatasetBase(
            root=path_zoo_root,
            layer_lst=layer_lst,
            epoch_lst=epoch_lst,
            mode="checkpoint",
            task="reconstruction",  # "reconstruction" (x->x), "sequence_prediction" (x^i -> x^i+1),
            use_bias=True,
            train_val_test="train",  # determines whcih dataset split to use
            ds_split=[0.7, 0.15, 0.15],  #
            max_samples=None,
            weight_threshold=5,
            filter_function=None,  # gets sample path as argument and returns True if model needs to be filtered out
            property_keys=property_keys,
            num_threads=1,
            verbosity=0,
            shuffle_path=True,
    )
    # valset
    valset = ModelDatasetBase(
            root=path_zoo_root,
            layer_lst=layer_lst,
            epoch_lst=epoch_lst,
            mode="checkpoint",
            task="reconstruction",  # "reconstruction" (x->x), "sequence_prediction" (x^i -> x^i+1),
            use_bias=True,
            train_val_test="val",  # determines whcih dataset split to use
            ds_split=[0.7, 0.15, 0.15],  #
            max_samples=None,
            weight_threshold=5,
            filter_function=None,  # gets sample path as argument and returns True if model needs to be filtered out
            property_keys=property_keys,
            num_threads=1,
            verbosity=0,
            shuffle_path=True,
    )
    # testset
    testset = ModelDatasetBase(
            root=path_zoo_root,
            layer_lst=layer_lst,
            epoch_lst=epoch_lst,
            mode="checkpoint",
            task="reconstruction",  # "reconstruction" (x->x), "sequence_prediction" (x^i -> x^i+1),
            use_bias=True,
            train_val_test="test",  # determines whcih dataset split to use
            ds_split=[0.7, 0.15, 0.15],  #
            max_samples=None,
            weight_threshold=5,
            filter_function=None,  # gets sample path as argument and returns True if model needs to be filtered out
            property_keys=property_keys,
            num_threads=1,
            verbosity=0,
            shuffle_path=True,
    )
    # put in dictionary
    dataset = {
        "trainset": trainset,
        "valset": valset,
        "testset": testset,
    }

    return dataset

def torch_to_haiku(pytorch_params):
    """
    Convert PyTorch parameters to Haiku parameters with correct nomenclature.
    Args:
        pytorch_params (collections.OrderedDict): PyTorch parameters.
    Returns:
        haiku_params (dict): Haiku parameters.
    """
    haiku_params = {}
    for k, v in pytorch_params.items():
        module, param_type = k.rsplit('.',1)
        module_name, idx = module.split('module_list.',1)

        if int(idx) in [0, 3, 6]:  # For conv2d layers
            layer_name = f'conv2d_{idx}'
            transpose_axes = (2, 3, 1, 0) if param_type == 'weight' else (0, )  # Transpose weights for conv layers
        elif int(idx) in [9, 11]:  # For linear (fully connected) layers
            layer_name = f'linear_{idx}'
            transpose_axes = (1, 0) if param_type == 'weight' else (0, )  # Transpose weights for linear layers

        # Define parameter type
        param_type = 'w' if param_type == 'weight' else 'b'

        # Create a new dict if module doesn't exist yet
        if layer_name not in haiku_params:
            haiku_params[layer_name] = {}

        # Set parameter
        haiku_params[layer_name][param_type] = jnp.transpose(v.numpy(), transpose_axes)

    return haiku_params

def properties_to_jax(properties:dict):
    """Convert properties lists to jnp arrays and encode classes as ints"""
    
    #TODO: pretty hard-coded now
    continuous_keys = ["test_acc", "training_iteration", "ggap", "model::dropout", "optim::lr", "optim::wd"]
    def encode_discrete_properties(properties):
        # First, map the distinct values to unique indices
        unique_properties, encoded_properties = np.unique(properties, return_inverse=True)
        return encoded_properties
    
    properties_jax = {}
    for k,v in properties.items():
        if k in continuous_keys:
            properties_jax[k] = jnp.array(v)
        else:
            properties_jax[k] = jnp.array(encode_discrete_properties(v))
    return properties_jax


def load_modelzoo(path:str, task:str=None, epochs:list=[5,15,25,50]):
    path = Path(path)
    data = load_dataset_from_path(path, epochs)
    
    train_data = [torch_to_haiku(tensor) for tensor in data['trainset'].data_in] # list of haiku param dicts
    train_labels = properties_to_jax(data['trainset'].properties)  # dictionary of labels
    
    val_data = [torch_to_haiku(tensor) for tensor in data['valset'].data_in] # list of haiku param dicts
    val_labels = properties_to_jax(data['valset'].properties)  # dictionary of labels
    
    test_data = [torch_to_haiku(tensor) for tensor in data['testset'].data_in] # list of haiku param dicts
    test_labels = properties_to_jax(data['testset'].properties)  # dictionary of labels
    
    if task is not None:
        train_labels = train_labels[task].reshape(-1,1)
        val_labels = val_labels[task].reshape(-1,1)
        test_labels = test_labels[task].reshape(-1,1)
    
    return train_data, train_labels, val_data, val_labels, test_data, test_labels