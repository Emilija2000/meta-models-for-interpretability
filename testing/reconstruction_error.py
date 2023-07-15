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
from pretraining import process_batch, MWMLossMSE
from chunking import preprocessing


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--train_data_dir',type=str,default='/rds/user/ed614/hpc-work/model_zoo_datasets/mnist_smallCNN_fixed_zoo')
    parser.add_argument('--test_data_dir',type=str,default='/rds/user/ed614/hpc-work/model_zoo_datasets/downstream_droppedcls_mnist_smallCNN_fixed_zoo')
    parser.add_argument('--pretrained_path',type=str,default='checkpoints/0_masked_chunked_indicators_64/1685836846.287388/36180')
    parser.add_argument('--mask_single',type=bool,default=False)
    parser.add_argument('--model_size',type=int,help='MetaModelClassifier model_size parameter',default=4*32)
    parser.add_argument('--num_layers',type=int,help='num of transformer layers',default=12)
    parser.add_argument('--num_heads',type=int,help='num of MHA heads',default=8)
    parser.add_argument('--chunk_size',type=int,help='meta model chunk size',default=128)
    parser.add_argument('--layerwise',action='store_true')
    parser.add_argument('--bs',type=int,default=32)
    parser.add_argument('--name',type=str,default='baseline')
    
    args = parser.parse_args()
    rng = random.PRNGKey(0)
    
    train_inputs, train_labels = load_nets(n=500, data_dir=args.train_data_dir, flatten=False, num_checkpoints=1)
    test_inputs, test_labels = load_nets(n=500, data_dir=args.test_data_dir, flatten=False, num_checkpoints=1)
    
    print("loaded data")

    model_config = ModelConfig(
        model_size=args.model_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=0.0,
        use_embedding=True,
    )
    model = create_meta_model(model_config)
    loss_fcn = MWMLossMSE(model.apply, non_masked=False)
    
    # load pretrained metamodel
    dummy_input,_,_,_ = process_batch(random.PRNGKey(0), train_inputs[:args.bs], 0,
                                    mask_prob=0.2, chunk_size=args.chunk_size, 
                                    mask_individual=args.mask_single, 
                                    mask_indicators=True,
                                    resample_zeromasks=False,
                                    layerwise=args.layerwise)
    rng,subkey = random.split(rng)
    params = model.init(subkey,utils.tree_stack(dummy_input), is_training=True)
    params = load_pretrained_meta_model_parameters(params, args.pretrained_path)
    
    # data iterators
    def data_iterator(masked_inputs:jnp.ndarray, inputs: jnp.ndarray, positions:jnp.ndarray,
                  batchsize: int = 1048, skip_last: bool = False):
        """Iterate over the data in batches."""
        for i in range(0, len(inputs), batchsize):
            if skip_last and i + batchsize > len(inputs):
                break
            yield (masked_inputs[i:i+batchsize],
                    inputs[i:i + batchsize], 
                    positions[i:i + batchsize])
            
    rng,subkey = random.split(rng)
    train_in, train_out, train_pos,_ = process_batch(subkey, train_inputs, mask_token=0,
                                                    mask_prob=0.2, chunk_size=args.chunk_size, 
                                                    mask_individual=args.mask_single, 
                                                    mask_indicators=True,
                                                    resample_zeromasks=False,
                                                    layerwise=args.layerwise)
    train_iterator = data_iterator(train_in, train_out, train_pos, batchsize=args.bs,skip_last=True)
    rng,subkey = random.split(rng)
    test_in, test_out, test_pos,_ = process_batch(subkey, test_inputs, mask_token=0,
                                                    mask_prob=0.2, chunk_size=args.chunk_size, 
                                                    mask_individual=args.mask_single, 
                                                    mask_indicators=True,
                                                    resample_zeromasks=False,
                                                    layerwise=args.layerwise)
    test_iterator = data_iterator(test_in, test_out, test_pos, batchsize=args.bs,skip_last=True)
    
    # check reconstruction r2 for training and test sets
    r2_train = []
    for masked_ins,target,positions in train_iterator:
        
        predicted = model.apply(params, subkey, masked_ins, False)
        l = loss_fcn.r2_score(predicted, target,positions)
        r2_train.append(l)
    print('R2 on training dataset: ',np.mean(r2_train))
    
    r2_test = []
    for masked_ins,target,positions in test_iterator:
        
        predicted = model.apply(params, subkey, masked_ins, False)
        l = loss_fcn.r2_score(predicted, target,positions)
        r2_test.append(l)
    print('R2 on testing dataset: ',np.mean(r2_test))
        