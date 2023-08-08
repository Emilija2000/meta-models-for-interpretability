import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import optax
from jax.typing import ArrayLike
from typing import Tuple, List, Iterator
import time
from math import ceil

from jax.tree_util import tree_flatten
from collections import defaultdict

from augmentations import augment_batch
from model_zoo_jax import load_nets, shuffle_data, load_data
from torchload import load_modelzoo

from model.meta_model import create_meta_model
from model.meta_model import MetaModelConfig as ModelConfig

from pretraining import MWMLossMSE, MWMLossMseNormalized, Updater, Logger, process_batch
from finetuning import load_pretrained_state

import os
import argparse

def split_data(data: list):
    split_index = int(len(data)*0.8)
    return (data[:split_index], 
            data[split_index:])

def flatten(x):
    return jax.flatten_util.ravel_pytree(x)[0]

def is_fine(params: dict):
    """Return false if std or mean is too high."""
    flat = flatten(params)
    if flat.std() > 5.0 or jnp.abs(flat.mean()) > 5.0:
        return False
    else:
        return True

def filter_data(data: List[dict]):
    """Given a list of net params, filter out those
    with very large means or stds."""
    f_data = [x for x in data if is_fine(x)]
    print(f"Filtered out {len(data) - len(f_data)} nets.\
          That's {100*(len(data) - len(f_data))/len(data):.2f}%.")
    return np.array(f_data)

def filter_poorly_trained(data:List[dict], accs:List[ArrayLike]):
    assert len(data) == len(accs)
    f_data = [x for x, y in zip(data, accs) if y>0.5]
    print(f"Filtered out {len(data) - len(f_data)} nets.\
          That's {100*(len(data) - len(f_data))/len(data):.2f}%.")
    return np.array(f_data)

def data_iterator(masked_inputs:jnp.ndarray, inputs: jnp.ndarray, positions:jnp.ndarray, non_positions:jnp.ndarray,
                  batchsize: int = 1048, skip_last: bool = False,variances:ArrayLike=None) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Iterate over the data in batches."""
    for i in range(0, len(inputs), batchsize):
        if skip_last and i + batchsize > len(inputs):
            break
        yield dict(masked_input = masked_inputs[i:i+batchsize],
                input=inputs[i:i + batchsize], 
                position=positions[i:i + batchsize],
                non_position=non_positions[i:i+batchsize],
                variances=variances)
        

def learning_rate_schedule(warmup_epochs, total_epochs, peak_lr, steps_per_epoch, decay_epochs=[0.4, 0.6, 0.8], decay_factor=0.2):
    """Custom learning rate schedule with warmup and decay."""
    def schedule(step):
        # Convert step to epoch
        epoch = step / steps_per_epoch

        # Warmup phase: linearly increase learning rate
        lr = jnp.where(epoch < warmup_epochs, peak_lr * (epoch / warmup_epochs), peak_lr)
        
        # Decay phase: decrease learning rate at 50% and 75% of total epochs
        for decay_epoch_ratio in decay_epochs:
            lr = jnp.where(epoch >= decay_epoch_ratio * total_epochs, lr * decay_factor, lr)
        return lr
    return schedule

def calculate_layer_stats(zoo, chunk_size):
    # Prepare a dictionary to store all parameter values for each layer
    layer_stats = defaultdict(list)
    layer_num_chunks = []
    flag=True
    #print("Model zoo layer shapes")
    for model_params in zoo:
        leaves, _ = jax.tree_util.tree_flatten(model_params)
        for leaf_id, leaf in enumerate(leaves):
            layer_stats[leaf_id].append(jnp.std(leaf.flatten()))
            if flag:
                original_length = len(leaf.flatten())
                layer_num_chunks.append(int(ceil(original_length/chunk_size)))
                #print(leaf_id, leaf.flatten().shape, layer_num_chunks[-1])
        flag = False
    # Calculate statistics for each layer
    avg_stats = []
    for layer_id, values in layer_stats.items():
        avg_stats.append(jnp.mean(jnp.array(values)))
    return jnp.square(jnp.array(avg_stats)), layer_num_chunks 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training run')
    # training parameters
    parser.add_argument('--lr', type=float, help='Learning rate', default=5e-5)
    parser.add_argument('--wd', type=float, help='Weight decay', default=0.0)
    parser.add_argument('--dropout',type=float,help='Meta-transformer dropout', default=0.0)
    parser.add_argument('--bs', type=int, help='Batch size', default=32)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=6000)
    #parser.add_argument('--adam_b1', type=float, help='Learning rate', default=0.1)
    #parser.add_argument('--adam_b2', type=float, help='Weight decay', default=0.001)
    #parser.add_argument('--adam_eps', type=float, help='Weight decay', default=1e-8)
    # meta-model
    parser.add_argument('--model_size',type=int,help='MetaModel model_size parameter',default=4*32)
    parser.add_argument('--num_layers',type=int,help='num of transformer layers',default=12)
    parser.add_argument('--num_heads',type=int,help='num of MHA heads',default=8)
    parser.add_argument('--chunk_size',type=int,help='meta model chunk size',default=64)
    parser.add_argument('--mask_prob',type=float,default=0.2)
    parser.add_argument('--mask_single',action='store_true',help='Mask each weight individually')
    #parser.add_argument('--mask_indicators',action='store_true',help='Include binary mask indicators to meta-model chunked input')
    parser.add_argument('--mask_indicators',type=bool, default=True,help='Include binary mask indicators to meta-model chunked input')
    #parser.add_argument('--include_nonmasked_loss',action='store_true')
    parser.add_argument('--include_nonmasked_loss',type=bool,default=False)
    # data
    parser.add_argument('--dataset_type',type=str,help='My dataset or external torch dataset. Values:myzoo or torchzoo',default='torchzoo')
    parser.add_argument('--data_dir',type=str,default='/rds/user/ed614/hpc-work/model_zoo_datasets/mnist_hyp_rand/tune_zoo_mnist_hyperparameter_10_random_seeds')
    parser.add_argument('--num_checkpoints',type=int,default=1)
    parser.add_argument('--num_networks',type=int,default=None)
    #parser.add_argument('--filter', action='store_true', help='Filter out high variance NN weights')
    parser.add_argument('--filter', type=bool, default=False, help='Filter out high variance NN weights')
    # augmentations
    #parser.add_argument('--augment', action='store_true', help='Use permutation augmentation')
    parser.add_argument('--augment', type=int, default=1, help='Use permutation augmentation')
    parser.add_argument('--num_augment',type=int,default=1)
    #logging
    #parser.add_argument('--use_wandb', action='store_true', help='Use wandb')
    parser.add_argument('--use_wandb', type=bool, default=True, help='Use wandb')
    parser.add_argument('--wandb_log_name', type=str, default="meta-transformer-pretraining-mnist")
    parser.add_argument('--log_interval',type=int, default=50)
    parser.add_argument('--seed',type=int, help='PRNG key seed',default=42)
    parser.add_argument('--exp', type=str, default="sweep")
    # continued training
    parser.add_argument('--pretrained_path',type=str,default=None)
    # type of chunking
    #parser.add_argument('--notlayerwise',action='store_true', help='turn off layerwise chunking')
    parser.add_argument('--notlayerwise',type=int,default=0, help='turn off layerwise chunking')
    #parser.add_argument('--layerind',action='store_true',help='indicators for layer type')
    parser.add_argument('--layerind',type=int, default=0,help='indicators for layer type')
    # for sweeps
    parser.add_argument('--max_runtime', type=int, help='Max runtime in minutes', default=np.inf)
    #parser.add_argument('--save_chkp',action='store_true')
    parser.add_argument('--save_chkp',type=int,default=0)
    args = parser.parse_args()
    
    rng = random.PRNGKey(args.seed)
    
    rng,subkey = random.split(rng)
    
    if args.mask_single:
        MASK_TOKEN=0
    else:
        #MASK_TOKEN = random.uniform(subkey,(args.chunk_size,),minval=-100, maxval=100)
        if args.layerind:
            MASK_TOKEN = jnp.zeros((args.chunk_size+4,))#TODO: hardcoded +4
        else:
            MASK_TOKEN = jnp.zeros((args.chunk_size,))
    
    '''    
    # Load model zoo checkpoints
    print(f"Loading model zoo: {args.data_dir}")
    inputs, labels = load_nets(n=args.num_networks, 
                                   data_dir=args.data_dir,
                                   flatten=False,
                                   num_checkpoints=args.num_checkpoints)
    
    #unpreprocess = preprocessing.get_unpreprocess(inputs[0], args.chunk_size)
    print("Loaded")
    # Filter (high variance and poorly trained networks)
    if args.filter:
        inputs = filter_poorly_trained(inputs, labels['test/acc'])
        inputs = filter_data(inputs)
        
    # Shuffle checkpoints before splitting
    rng, subkey = random.split(rng)
    filtered_inputs, _ = shuffle_data(subkey,inputs,np.zeros(len(inputs)),chunks=args.num_checkpoints)
    print("Shuffled")
    train_inputs, val_inputs = split_data(filtered_inputs)
    '''
    
    # Load data
    if args.dataset_type == 'myzoo':
        rng,subkey = random.split(rng)
        train_inputs, _, val_inputs, _, test_inputs, _ = load_data(subkey, args.data_dir, None,args.num_networks,args.num_checkpoints, is_filter=args.filter)
    else:
        train_inputs, _, val_inputs, _, test_inputs, _ = load_modelzoo(args.data_dir, None, epochs=list(range(0,51,50//args.num_checkpoints))[1:])

        # Keep a subset for finetuning/baseline
        splitkey = random.PRNGKey(123)
        train_inputs, _ = shuffle_data(splitkey, train_inputs, jnp.ones(len(train_inputs)))
        train_inputs = train_inputs[:6000]#[:int(0.8*len(train_inputs))] #6000 
        val_inputs, _ = shuffle_data(splitkey, val_inputs, jnp.ones(len(val_inputs)))
        val_inputs =  val_inputs[:1000]#[:int(0.65*len(val_inputs))] #1000
        test_inputs, _ = shuffle_data(splitkey, test_inputs, jnp.ones(len(test_inputs)))
        test_inputs = test_inputs[:500]#[:int(0.65*len(test_inputs))] #500
        
        if args.num_networks is not None and args.num_networks < len(train_inputs):
            train_inputs = train_inputs[:args.num_networks]


    steps_per_epoch = len(train_inputs) // args.bs
    print()
    print(f"Number of training examples: {len(train_inputs)}.")
    print("Steps per epoch:", steps_per_epoch)
    print("Total number of steps:", steps_per_epoch * args.epochs)
    print()
    
    # layer stats
    layer_vars, chunks_per_layer = calculate_layer_stats(train_inputs, args.chunk_size)
    print(layer_vars, chunks_per_layer)
    chunk_vars = [v for v, c in zip(layer_vars, chunks_per_layer) for i in range(c)]
    chunk_vars = jnp.array(chunk_vars)
    # calculating chunk stats - loss normalization
    #chunked_ins,_,_,_ = process_batch(random.PRNGKey(0), train_inputs[:1000], 0,
    #                                mask_prob=0, chunk_size=args.chunk_size, 
    #                                mask_individual=args.mask_single, 
    #                                mask_indicators=False,resample_zeromasks=False)
    #chunked_ins=jnp.transpose(chunked_ins,axes=[1,0,2])
    #means = jnp.mean(jnp.reshape(chunked_ins,(chunked_ins.shape[0],-1)),axis=1)
    #chunk_vars = jnp.std(jnp.reshape(chunked_ins,(chunked_ins.shape[0],-1)),axis=1)
    #chunk_vars = jnp.ones(chunk_vars.shape)
    #chunk_vars = jnp.square(chunk_vars)
    print("Chunk variances calculated\n")
    #print(chunk_vars)

    # model
    model_config = ModelConfig(
        model_size=args.model_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout,
        use_embedding=True,
    )

    lr_schedule = learning_rate_schedule(10, args.epochs, args.lr,steps_per_epoch)

    # Initialization
    model = create_meta_model(model_config)
    if args.notlayerwise:
        loss_fcn = MWMLossMSE(model.apply, non_masked=args.include_nonmasked_loss)
    else:
        loss_fcn = MWMLossMseNormalized(model.apply, non_masked=args.include_nonmasked_loss)
    #loss_fcn = MWMLossCosine(model.apply, non_masked=args.include_nonmasked_loss)
    opt = optax.adamw(learning_rate=lr_schedule, weight_decay=args.wd)
    updater = Updater(opt=opt, evaluator=loss_fcn, model_init=model.init)
    
    rng, subkey = random.split(rng)
        
    dummy_input,_,_,_ = process_batch(jax.random.PRNGKey(0), train_inputs[:args.bs], 0,mask_prob=args.mask_prob, chunk_size=args.chunk_size,
                                    mask_individual=args.mask_single, mask_indicators=args.mask_indicators,
                                    layerwise=not(args.notlayerwise),
                                    layerind=args.layerind)
    state = updater.init_params(subkey, x=dummy_input) #
    
    if args.pretrained_path is not None:
        state = load_pretrained_state(state, args.pretrained_path)

    print("Number of parameters:", sum(x.size for x in jax.tree_util.tree_leaves(state.params)) / 1e6, "Million")
    
    # logger
    logger = Logger(name = args.wandb_log_name,
                    config={
                    "seed":args.seed,
                    "exp": args.exp,
                    "dataset": os.path.basename(args.data_dir),
                    "lr": args.lr,
                    "weight_decay": args.wd,
                    "batchsize": args.bs,
                    "num_epochs": args.epochs,
                    "dropout": args.dropout,
                    "model_size":args.model_size,
                    "chunk_size":args.chunk_size,
                    "num_layers":args.num_layers,
                    "augment":args.augment,
                    "num_networks":args.num_networks,
                    "mask_prob":args.mask_prob
                    },
                    log_wandb = args.use_wandb,
                    save_checkpoints=args.save_chkp,
                    save_interval=10,
                    log_interval=args.log_interval,
                    checkpoint_dir=os.path.join('checkpoints',args.exp,str(time.time())))
    logger.init(is_save_config=True)

    # Training loop
    best_loss = 100000
    best = None
    start = time.time()
    max_runtime_reached = False
    for epoch in range(args.epochs):
        rng,subkey = random.split(rng)
        
        if args.augment:
            images,_ = augment_batch(subkey,train_inputs,np.zeros(len(train_inputs)),num_p=args.num_augment,keep_original=False)
        else:
            images = train_inputs
        rng, subkey = random.split(rng)
        images, _ = shuffle_data(subkey, images, np.zeros(len(images)))
        rng, subkey = random.split(rng)
        masked_ins, masked_labels, positions, non_masked_positions = process_batch(subkey, images, MASK_TOKEN, 
                                                      mask_prob=args.mask_prob,
                                                      chunk_size=args.chunk_size,
                                                      mask_individual=args.mask_single, 
                                                      mask_indicators=args.mask_indicators,
                                                      layerwise=not(args.notlayerwise),
                                                      layerind=args.layerind)
        batches = data_iterator(masked_ins, masked_labels, positions,non_masked_positions, batchsize=args.bs, skip_last=True,variances=chunk_vars)

        train_all_loss = []
        for it, batch in enumerate(batches):
            state, train_metrics = updater.train_step(state, (batch['masked_input'],batch['input'],batch['position'],batch['non_position'],batch['variances']))
            logger.log(state, train_metrics)
            train_all_loss.append(train_metrics['train/loss'].item())
            
            if time.time() - start > args.max_runtime * 60:
                print("=======================================")
                print("Max runtime reached. Stopping training.")
                max_runtime_reached = True
                break
        train_metrics = {'train/avg_loss':np.mean(train_all_loss)}
            
        # Validate every epoch
        rng, subkey = random.split(rng)
        masked_ins, masked_labels, positions,non_masked_positions = process_batch(subkey, val_inputs, MASK_TOKEN, 
                                                      #mask_prob=args.mask_prob,
                                                      mask_prob=0.2,
                                                      chunk_size=args.chunk_size, 
                                                      mask_individual=args.mask_single, 
                                                      mask_indicators=args.mask_indicators,
                                                      layerwise=not(args.notlayerwise),
                                                      layerind=args.layerind)
        batches = data_iterator(masked_ins, masked_labels, positions,non_masked_positions, batchsize=args.bs, skip_last=True,variances=chunk_vars)
        val_all_loss = []
        for it, batch in enumerate(batches):
            state, val_metrics = updater.val_step(state, (batch['masked_input'],batch['input'],batch['position'],batch['non_position'],batch['variances']))
            #logger.log(state, val_metrics)
            val_all_loss.append(val_metrics['val/loss'].item())
        val_metrics = {'val/avg_loss':np.mean(val_all_loss)}
            
        logger.log(state, train_metrics, val_metrics)
        
        if val_metrics['val/avg_loss'] <best_loss:
            best_loss = val_metrics['val/avg_loss']
            best = state
        
        if max_runtime_reached:
            break
        
    logger.save_checkpoint(best, train_metrics, val_metrics)
        
    # Evaluate reconstruction error on test set
    test_in, test_out, test_pos,non_pos = process_batch(subkey, test_inputs, mask_token=0,
                                                    mask_prob=0.2, 
                                                    chunk_size=args.chunk_size, 
                                                    mask_individual=args.mask_single, 
                                                    mask_indicators=args.mask_indicators,
                                                    layerwise=not(args.notlayerwise),
                                                    layerind=args.layerind)
    test_iterator = data_iterator(test_in, test_out, test_pos,non_pos, batchsize=args.bs,skip_last=True)
    predictions = None
    for masked_ins,target,positions,_,_ in test_iterator:
        
        predicted = model.apply(state.params, subkey, masked_ins['masked_input'], False)
        if predictions is None:
            predictions = predicted
        else:
            predictions = jnp.concatenate([predictions, predicted],axis=0)
       
    loss_fcn = MWMLossMSE(model.apply, non_masked=False) 
    r2 = loss_fcn.r2_score(predictions, test_out,test_pos)
    print('R2 on test dataset: ',r2)
    logger.log(state,{'test/r2':r2.item()})
    