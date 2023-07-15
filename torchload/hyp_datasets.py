"""
Short mostly hard-coded functions that read data from NN datasets published
with https://arxiv.org/pdf/2002.11448.pdf in a format that works our meta-model (pre-chunking)
"""

import ast
import csv
import gzip
import jax.numpy as jnp
import numpy as np
import os


def load_csv_gz_as_dict(filename):
    with gzip.open(filename, 'rt') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Get column names from first row
        columns = {header: [] for header in headers}  # Initialize a dict with column names

        numerical_keys = []
        for row in reader:
            for header, value in zip(headers, row):
                try:
                    columns[header].append(float(value)) # convert to float before appending
                    if header not in numerical_keys:
                        numerical_keys.append(header)
                except:
                    columns[header].append(value)
    # Convert lists to jnp arrays
    for key, values in columns.items():
        if key in numerical_keys:
            columns[key] = jnp.array(values)

    return columns

def parse_params_file(filename):
    "Read parameter shapes"
    shapes = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            name, start, end, _, shape = row
            name = name.replace('dense','linear')
            name = name.replace('bias','b')
            name = name.replace('kernel','w')
            layer_name, param_name = name.rsplit('/',1)  # split name into layer and parameter
            param_name = param_name.split(':')[0]  # remove ":0" from parameter names
            start, end, shape = int(start), int(end), ast.literal_eval(shape)

            if layer_name not in shapes:
                shapes[layer_name] = {}

            shapes[layer_name][param_name] = (start, end, shape)
    return shapes

def vector_to_params(vector, shapes):
    """Reshape a flattened numpy vector of NN parameters into params dict with layers of shape in shapes"""
    params = {}
    for layer_name, layer_shapes in shapes.items():
        params[layer_name] = {}
        for param_name, (start, end, shape) in layer_shapes.items():
            params[layer_name][param_name] = jnp.reshape(vector[start:end], shape)
    return params

def read_params(params_path, shapes_path):
    '''Read NN parameter list'''
    param_shapes = parse_params_file(shapes_path)
    data = np.load(params_path)
    params_list = [vector_to_params(vector,param_shapes) for vector in data]
    return params_list

def load_dataset(dataset_path):
    params_list = read_params(
        params_path=os.path.join(dataset_path,'weights.npy'),
        shapes_path=os.path.join(dataset_path,'layout.csv')
        )
    
    labels = load_csv_gz_as_dict(os.path.join(dataset_path,'metrics.csv.gz'))
    return params_list,labels
    
    
if __name__=='__main__':
    import jax
    
    params_list = read_params(
        params_path='/rds/user/ed614/hpc-work/model_zoo_datasets/hyp/mnist/weights.npy',
        shapes_path='/rds/user/ed614/hpc-work/model_zoo_datasets/hyp/mnist/layout.csv'
        )
    print(len(params_list))
    print(jax.tree_util.tree_map(lambda x:x.shape, params_list[0]))

    labels = load_csv_gz_as_dict('/rds/user/ed614/hpc-work/model_zoo_datasets/hyp/mnist/metrics.csv.gz')
    print(labels.keys())
    print([labels[k][0] for k in labels.keys()])
    print(labels['train_accuracy'].shape)
