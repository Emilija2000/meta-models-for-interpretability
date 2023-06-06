import argparse
import functools
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.tree_util import tree_flatten
from jax import random,vmap,nn
import numpy as np

# meta model part
from model_zoo_jax import load_nets
from meta_transformer import utils, preprocessing
from meta_transformer.meta_model import create_meta_model
from meta_transformer.meta_model import MetaModelConfig as ModelConfig
from finetuning import load_pretrained_meta_model_parameters
from pretraining import process_batch

# MNIST part
from model_zoo_jax.datasets.nontorch_dropclassdataset import load_dataset,drop_class_from_datasets, get_dataloaders
from model_zoo_jax.models import get_model
from model_zoo_jax.config import sample_parameters

from collections import defaultdict

def calculate_layer_statistics(zoo):
    # Prepare a dictionary to store all parameter values for each layer
    layer_values = defaultdict(list)

    for model_params in zoo:
        leaves, _ = tree_flatten(model_params)
        for leaf_id, leaf in enumerate(leaves):
            layer_values[leaf_id].append(leaf)

    # Calculate statistics for each layer
    layer_stats = {}
    for layer_id, values in layer_values.items():
        # Concatenate all parameter values for this layer across all models
        all_values = jnp.concatenate(values)
        layer_stats[layer_id] = {'mean': float(jnp.mean(all_values)), 'std': float(jnp.std(all_values))}

    return layer_stats

def visualize_layer_statistics(layer_stats):
    for layer_name, stats in layer_stats.items():
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f'Statistics for layer: {layer_name}')

        ax1.plot(stats['means'])
        ax1.set_title('Means')

        ax2.plot(stats['stds'])
        ax2.set_title('Standard deviations')

        plt.show()
        
def metric_from_logits(logits, targets):
    """expects index targets, not one-hot"""
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == targets)

def loss_from_logits(logits, targets):
    """targets are index labels"""
    targets = nn.one_hot(targets, 9)
    return -jnp.sum(targets * nn.log_softmax(logits, axis=-1), axis=-1).mean()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--data_dir',type=str,default='/rds/user/ed614/hpc-work/model_zoo_datasets/downstream_droppedcls_mnist_smallCNN_fixed_zoo')
    parser.add_argument('--pretrained_path',type=str,default='checkpoints/0_masked_chunked_indicators_64/1685836846.287388/36180')
    parser.add_argument('--mask_single',type=bool,default=False)
    parser.add_argument('--chunk_size',type=int,default=64)
    args = parser.parse_args()
    rng = random.PRNGKey(0)
    
    inputs, labels = load_nets(n=1000, data_dir=args.data_dir, flatten=False, num_checkpoints=1)
    class_dropped = labels['class_dropped']
    
    # stats
    layer_stats = calculate_layer_statistics(inputs)
    import json
    with open("layer_stats.json", "w") as fp:
        json.dump(layer_stats , fp)
    #print(layer_stats)
    #visualize_layer_statistics(layer_stats)
    
    # model def
    unpreprocess = preprocessing.get_unpreprocess(inputs[0], args.chunk_size,verbose=True)
    model_config = ModelConfig(
        model_size=64,
        num_heads=8,
        num_layers=12,
        dropout_rate=0.0,
        use_embedding=True,
    )
    model = create_meta_model(model_config)
    
    # load pretrained metamodel
    dummy_input,_,_,_ = process_batch(random.PRNGKey(0), inputs[:32], 0,
                                    mask_prob=0, chunk_size=args.chunk_size, 
                                    mask_individual=args.mask_single, 
                                    mask_indicators=True)
    rng,subkey = random.split(rng)
    params = model.init(subkey,utils.tree_stack(dummy_input), is_training=True)
    params = load_pretrained_meta_model_parameters(params, args.pretrained_path)
    
    # load mnist data
    datasets_mnist = load_dataset('MNIST')
    datasets = drop_class_from_datasets(datasets_mnist, class_dropped[0])
    dataloaders = get_dataloaders(datasets, 32)
    
    # get model fcn
    _,mnist_config = sample_parameters(random.PRNGKey(0), 'MNIST',
                              class_dropped=class_dropped[0], 
                              activation='leakyrelu',model_name='smallCNN')
    model_mnist,is_batch = get_model(mnist_config)
    if not(is_batch):
        batch_apply = vmap(model_mnist.apply, in_axes=(None,None,0,None),axis_name='batch')
        init_x = datasets['train'][0][0]
    else:
        batch_apply = model_mnist.apply
        init_x = next(iter(dataloaders['train']))[0]
        
    # check loss and acc for both nets
    for ni in range(5):
        # predict some missing weights
        rng,subkey = random.split(rng)
        masked_ins, net_original, positions, non_masked_positions = process_batch(subkey, inputs[:32], 0,
                                        mask_prob=0.15, chunk_size=args.chunk_size, 
                                        mask_individual=args.mask_single, 
                                        mask_indicators=True)
        rng,subkey = random.split(rng)
        predicted = model.apply(params, subkey, masked_ins, False)
        
        # network after meta model
        net2 = jnp.multiply(1-positions, net_original) + jnp.multiply(positions, predicted)
        sth = positions+non_masked_positions
        sth = sth.astype(bool)
        net2 = jnp.reshape(net2[sth],(32,-1,args.chunk_size))
        
        print(jnp.where(positions[ni])[0])
        
        datasets = drop_class_from_datasets(datasets_mnist, class_dropped[ni])
        dataloaders = get_dataloaders(datasets, 32)
        
        # network before and after metamodel
        net1_params = inputs[ni]
        net2_params = unpreprocess(net2[ni])
        
        test_acc_1 = []
        test_loss_1 = []
        test_acc_2 = []
        test_loss_2 = []
        for i,batch in enumerate(dataloaders['test']):
            images = batch[0]
            targets = batch[1]
            
            logits1 = batch_apply(net1_params, random.PRNGKey(0), images, False)
            test_acc_1.append(metric_from_logits(logits1,targets).item())
            test_loss_1.append(loss_from_logits(logits1,targets).item())
            
            logits2 = batch_apply(net2_params, random.PRNGKey(0), images, False)
            test_acc_2.append(metric_from_logits(logits2,targets).item())
            test_loss_2.append(loss_from_logits(logits2,targets).item())
        
        test_metrics_1 = {'test/acc':np.mean(test_acc_1), 'test/loss':np.mean(test_loss_1)}
        test_metrics_2 = {'test/acc':np.mean(test_acc_2), 'test/loss':np.mean(test_loss_2)}
        
        print(test_metrics_1)
        print(test_metrics_2)
