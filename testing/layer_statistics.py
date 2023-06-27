import argparse
import functools
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.tree_util import tree_flatten
from jax import random,vmap,nn
import numpy as np

# meta model part
from model_zoo_jax import load_nets
from meta_transformer import utils
from meta_transformer.meta_model import create_meta_model
from meta_transformer.meta_model import MetaModelConfig as ModelConfig
from finetuning import load_pretrained_meta_model_parameters
from pretraining import process_batch, MWMLossMseNormalized
from chunking import preprocessing

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

def generate_matrix(key, means, stds, length):
    rows = []
    for i in range(len(means)):
        key, subkey = random.split(key)
        row = random.normal(subkey, (length,))
        row = row * stds[i] + means[i]
        rows.append(row)
    matrix = jnp.stack(rows)
    return matrix

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--data_dir',type=str,default='/rds/user/ed614/hpc-work/model_zoo_datasets/downstream_droppedcls_mnist_smallCNN_fixed_zoo')
    parser.add_argument('--pretrained_path',type=str,default='checkpoints/0_masked_chunked_indicators_64/1685836846.287388/36180')
    parser.add_argument('--mask_single',type=bool,default=False)
    parser.add_argument('--model_size',type=int,help='MetaModelClassifier model_size parameter',default=4*32)
    parser.add_argument('--num_layers',type=int,help='num of transformer layers',default=12)
    parser.add_argument('--num_heads',type=int,help='num of MHA heads',default=8)
    parser.add_argument('--chunk_size',type=int,help='meta model chunk size',default=128)
    parser.add_argument('--image_data',type=str,default='MNIST')
    parser.add_argument('--base_arch',type=str,default='smallCNN')
    parser.add_argument('--layerwise',action='store_true')
    parser.add_argument('--bs',type=int,default=32)
    
    args = parser.parse_args()
    rng = random.PRNGKey(0)
    
    inputs, labels = load_nets(n=1000, data_dir=args.data_dir, flatten=False, num_checkpoints=1)
    class_dropped = labels['class_dropped']
    
    print("loaded")
    
    # stats
    layer_stats = calculate_layer_statistics(inputs)
    #import json
    #with open("layer_stats.json", "w") as fp:
    #    json.dump(layer_stats , fp)
    print(layer_stats)
    #visualize_layer_statistics(layer_stats)
    
    # chunk stats
    chunked_ins,_,_,_ = process_batch(random.PRNGKey(0), inputs, 0,
                                    mask_prob=0, chunk_size=args.chunk_size, 
                                    mask_individual=args.mask_single, 
                                    mask_indicators=False,
                                    resample_zeromasks=False,
                                    layerwise=args.layerwise)
    print(chunked_ins.shape)
    chunked_ins=jnp.transpose(chunked_ins,axes=[1,0,2])
    means = jnp.mean(jnp.reshape(chunked_ins,(chunked_ins.shape[0],-1)),axis=1)
    stds = jnp.std(jnp.reshape(chunked_ins,(chunked_ins.shape[0],-1)),axis=1)
    print(means.shape)
    chunk_stats = {str(i):{"mean":means[i].item(),"std":stds[i].item()} for i in range(means.shape[0])}
    #with open("chunk_stats.json", "w") as fp:
    #    json.dump(chunk_stats , fp)
    
    # model def
    if args.layerwise:
        unpreprocess = preprocessing.get_unpreprocess_layerwise(inputs[0], args.chunk_size,verbose=True)
    else:
        unpreprocess = preprocessing.get_unpreprocess(inputs[0], args.chunk_size,verbose=True)
    model_config = ModelConfig(
        model_size=args.model_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=0.0,
        use_embedding=True,
    )
    model = create_meta_model(model_config)
    pretrain_loss_fcn = MWMLossMseNormalized(model.apply, non_masked=False)
    
    # load pretrained metamodel
    dummy_input,_,_,_ = process_batch(random.PRNGKey(0), inputs[:args.bs], 0,
                                    mask_prob=0, chunk_size=args.chunk_size, 
                                    mask_individual=args.mask_single, 
                                    mask_indicators=True,
                                    resample_zeromasks=False,
                                    layerwise=args.layerwise)
    rng,subkey = random.split(rng)
    params = model.init(subkey,utils.tree_stack(dummy_input), is_training=True)
    params = load_pretrained_meta_model_parameters(params, args.pretrained_path)
    
    # load mnist data
    datasets_mnist = load_dataset(args.image_data)
    datasets = drop_class_from_datasets(datasets_mnist, class_dropped[0])
    dataloaders = get_dataloaders(datasets, args.bs)
    
    # get model fcn
    _,mnist_config = sample_parameters(random.PRNGKey(0), args.image_data,
                              class_dropped=class_dropped[0], 
                              activation='leakyrelu',model_name=args.base_arch)
    model_mnist,is_batch = get_model(mnist_config)
    if not(is_batch):
        batch_apply = vmap(model_mnist.apply, in_axes=(None,None,0,None),axis_name='batch')
        init_x = datasets['train'][0][0]
    else:
        batch_apply = model_mnist.apply
        init_x = next(iter(dataloaders['train']))[0]    
        
    print('prepared mnist dataloaders')
        
    # metrics to test
    avg_loss_random = []
    avg_loss_randN = []
    avg_loss_pretrained = []
    
    img_loss_random = []
    img_loss_randN = []
    img_loss_pretrained = []
    
    img_acc_random = []
    img_acc_randN = []
    img_acc_pretrained = []
    
    img_loss_original = []
    img_acc_original = []
    
    # check reconstruction loss and original dataset acc for both nets
    for ni in range(20):
        print(ni)
        # predict some missing weights
        rng,subkey = random.split(rng)
        masked_ins, net_original, positions, non_masked_positions = process_batch(subkey, inputs[ni*args.bs:(ni+1)*args.bs], 0,
                                        mask_prob=0.15, chunk_size=args.chunk_size, 
                                        mask_individual=args.mask_single, 
                                        mask_indicators=True,
                                        resample_zeromasks=False,
                                        layerwise=args.layerwise)
        rng,subkey = random.split(rng)
        predicted = model.apply(params, subkey, masked_ins, False)
        
        # network after meta model
        net2 = jnp.multiply(1-positions, net_original) + jnp.multiply(positions, predicted)
        sth = positions+non_masked_positions
        sth = sth.astype(bool)
        net2 = jnp.reshape(net2[sth],(args.bs,-1,args.chunk_size))
        
        # network with random predictions generated from stats
        rand_stats = []
        for gen in range(args.bs):
            rng,subkey = random.split(rng)
            rand_stats.append(generate_matrix(subkey, means,stds,args.chunk_size+1))
        rand_stats = jnp.stack(rand_stats)
        
        net3 = jnp.multiply(1-positions, net_original) + jnp.multiply(positions, rand_stats)
        sth = positions+non_masked_positions
        sth = sth.astype(bool)
        net3 = jnp.reshape(net3[sth],(args.bs,-1,args.chunk_size))
        
        # network with totally random predictions
        total_rand_stats = []
        for gen in range(args.bs):
            rng,subkey = random.split(rng)
            total_rand_stats.append(generate_matrix(subkey, jnp.zeros(means.shape),jnp.ones(means.shape),args.chunk_size+1))
        total_rand_stats = jnp.stack(total_rand_stats)
        
        net4 = jnp.multiply(1-positions, net_original) + jnp.multiply(positions, total_rand_stats)
        sth = positions+non_masked_positions
        sth = sth.astype(bool)
        net4 = jnp.reshape(net4[sth],(args.bs,-1,args.chunk_size))
        
        # pretrain loss for act prediction and for random
        l = pretrain_loss_fcn.masked_loss_fn(total_rand_stats, net_original, positions,jnp.square(stds))
        avg_loss_random.append(l.item())
        #print("Random pretrain loss ", l.item())
        l = pretrain_loss_fcn.masked_loss_fn(rand_stats, net_original, positions,jnp.square(stds))
        avg_loss_randN.append(l.item())
        #print("Random ~(mean,std) pretrain loss ", l.item())
        l = pretrain_loss_fcn.masked_loss_fn(predicted, net_original, positions,jnp.square(stds))
        avg_loss_pretrained.append(l.item())
        #print("Prediction pretrain loss ", l.item())
        
        
        datasets = drop_class_from_datasets(datasets_mnist, class_dropped[ni])
        dataloaders = get_dataloaders(datasets, 32)
        
        # network before and after metamodel
        net1_params = inputs[ni]
        net2_params = unpreprocess(net2[ni])
        net3_params = unpreprocess(net3[ni])
        net4_params = unpreprocess(net4[ni])
        
        test_acc_1 = []
        test_loss_1 = []
        test_acc_2 = []
        test_loss_2 = []
        test_acc_3 = []
        test_loss_3 = []
        test_acc_4 = []
        test_loss_4 = []
        for i,batch in enumerate(dataloaders['test'][:100]):
            images = batch[0]
            targets = batch[1]
            
            logits1 = batch_apply(net1_params, random.PRNGKey(0), images, False)
            test_acc_1.append(metric_from_logits(logits1,targets).item())
            test_loss_1.append(loss_from_logits(logits1,targets).item())
            
            logits2 = batch_apply(net2_params, random.PRNGKey(0), images, False)
            test_acc_2.append(metric_from_logits(logits2,targets).item())
            test_loss_2.append(loss_from_logits(logits2,targets).item())
            
            logits3 = batch_apply(net3_params, random.PRNGKey(0), images, False)
            test_acc_3.append(metric_from_logits(logits3,targets).item())
            test_loss_3.append(loss_from_logits(logits3,targets).item())
            
            logits4 = batch_apply(net4_params, random.PRNGKey(0), images, False)
            test_acc_4.append(metric_from_logits(logits4,targets).item())
            test_loss_4.append(loss_from_logits(logits4,targets).item())
        
        #test_metrics_1 = {'test/acc':np.mean(test_acc_1), 'test/loss':np.mean(test_loss_1)}
        #test_metrics_2 = {'test/acc':np.mean(test_acc_2), 'test/loss':np.mean(test_loss_2)}
        #test_metrics_3 = {'test/acc':np.mean(test_acc_3), 'test/loss':np.mean(test_loss_3)}
        #test_metrics_4 = {'test/acc':np.mean(test_acc_4), 'test/loss':np.mean(test_loss_4)}
        
        #print("Original: ", test_metrics_1)
        #print("Prediction: ", test_metrics_2)
        #print("Random ~(mean,var): ", test_metrics_3)
        #print("Random: ", test_metrics_4)
        #print()    
        
        img_acc_original.append(np.mean(test_acc_1))
        img_loss_original.append(np.mean(test_loss_1))
        img_acc_pretrained.append(np.mean(test_acc_2))
        img_loss_pretrained.append(np.mean(test_loss_2))
        img_acc_randN.append(np.mean(test_acc_3))
        img_loss_randN.append(np.mean(test_loss_3))
        img_acc_random.append(np.mean(test_acc_4))
        img_loss_random.append(np.mean(test_loss_4))
        
    
    print("Weight reconstruction loss")
    print("Random ", np.mean(avg_loss_random))
    print("Random ~(mean,std) ", np.mean(avg_loss_randN))
    print("Pretrained: ", np.mean(avg_loss_pretrained))
    print()
    print("Average image dataset performance")
    print('Original: {}, {}'.format(np.mean(img_acc_original),np.mean(img_loss_original)))
    print('Prediction: {}, {}'.format(np.mean(img_acc_pretrained),np.mean(img_loss_pretrained)))
    print('Random ~(mean,var): {}, {}'.format(np.mean(img_acc_randN),np.mean(img_loss_randN)))
    print('Random: {}, {}'.format(np.mean(img_acc_random),np.mean(img_loss_random)))
    
