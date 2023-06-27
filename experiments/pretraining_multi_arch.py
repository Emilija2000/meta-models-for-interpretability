import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import optax
from jax.typing import ArrayLike
from typing import Tuple, List, Iterator
import time

from meta_transformer import utils 
from augmentations import augment, augment_whole
from model_zoo_jax import load_nets, shuffle_data, load_multiple_datasets

from meta_transformer.meta_model import create_meta_model
from meta_transformer.meta_model import MetaModelConfig as ModelConfig

from pretraining import MWMLossMSE, Updater, Logger, process_batch

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
        

def learning_rate_schedule(warmup_epochs, total_epochs, peak_lr, steps_per_epoch, decay_epochs=[0.5, 0.75], decay_factor=0.1):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training run')
    # training parameters
    parser.add_argument('--lr', type=float, help='Learning rate', default=5e-5)
    parser.add_argument('--wd', type=float, help='Weight decay', default=1e-3)
    parser.add_argument('--dropout',type=float,help='Meta-transformer dropout', default=0.1)
    parser.add_argument('--bs', type=int, help='Batch size', default=32)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=500)
    # meta-model
    parser.add_argument('--model_size',type=int,help='MetaModel model_size parameter',default=4*32)
    parser.add_argument('--num_layers',type=int,help='num of transformer layers',default=12)
    parser.add_argument('--num_heads',type=int,help='num of MHA heads',default=8)
    parser.add_argument('--chunk_size',type=int,help='meta model chunk size',default=128)
    parser.add_argument('--mask_prob',type=float,default=0.2)
    parser.add_argument('--mask_single',action='store_true',help='Mask each weight individually')
    parser.add_argument('--mask_indicators',action='store_true',help='Include binary mask indicators to meta-model chunked input')
    #parser.add_argument('--mask_indicators',type=bool, default=True,help='Include binary mask indicators to meta-model chunked input')
    parser.add_argument('--include_nonmasked_loss',action='store_true')
    # data
    parser.add_argument('--num_checkpoints',type=int,default=1)
    parser.add_argument('--num_networks',type=int,default=None)
    parser.add_argument('--filter', action='store_true', help='Filter out high variance NN weights')
    #parser.add_argument('--filter', type=bool, default=True, help='Filter out high variance NN weights')
    # augmentations
    parser.add_argument('--augment', action='store_true', help='Use permutation augmentation')
    #parser.add_argument('--augment', type=bool, default=True, help='Use permutation augmentation')
    parser.add_argument('--num_augment',type=int,default=1)
    #logging
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb')
    #parser.add_argument('--use_wandb', type=bool, default=True, help='Use wandb')
    parser.add_argument('--wandb_log_name', type=str, default="meta-transformer-pretraining-fixed-mnist")
    parser.add_argument('--log_interval',type=int, default=50)
    parser.add_argument('--seed',type=int, help='PRNG key seed',default=42)
    parser.add_argument('--exp', type=str, default="sweep")
    args = parser.parse_args()
    
    rng = random.PRNGKey(args.seed)
    
    rng,subkey = random.split(rng)
    
    DIR1 = '/rds/user/ed614/hpc-work/model_zoo_datasets/mnist_smallCNN_fixed_zoo'
    DIR2 = '/rds/user/ed614/hpc-work/model_zoo_datasets/cifar10_lenet5_fixed_zoo'
    DIR3 = '/rds/user/ed614/hpc-work/model_zoo_datasets/mnist_smallCNN_random_zoo'
    
    if args.mask_single:
        MASK_TOKEN=0
    else:
        #MASK_TOKEN = random.uniform(subkey,(args.chunk_size,),minval=-100, maxval=100)
        MASK_TOKEN = jnp.zeros((args.chunk_size,))

    # Load model zoo checkpoints
    print(f"Loading model zoo")
    inputs, labels = load_multiple_datasets([DIR1, DIR2, DIR3],
                                            num_networks=args.num_networks,
                                            num_checkpoints=args.num_checkpoints,
                                            verbose=True,
                                            bs=args.bs) #load multiple of bs so we can batch augment and process
    
    
    # Filter (high variance and poorly trained networks)
    if args.filter:
        inputs = filter_poorly_trained(inputs, labels['test/acc'])
        inputs = filter_data(inputs)
        
    # Shuffle checkpoints before splitting
    rng, subkey = random.split(rng)
    filtered_inputs, _ = shuffle_data(subkey,inputs,np.zeros(len(inputs)),chunks=args.num_checkpoints*args.bs)
    
    train_inputs, val_inputs = split_data(filtered_inputs)

    steps_per_epoch = len(train_inputs) // args.bs
    print()
    print(f"Number of training examples: {len(train_inputs)}.")
    print("Steps per epoch:", steps_per_epoch)
    print("Total number of steps:", steps_per_epoch * args.epochs)
    print()

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
    loss_fcn = MWMLossMSE(model.apply, non_masked=args.include_nonmasked_loss)
    opt = optax.adamw(learning_rate=lr_schedule, weight_decay=args.wd)
    updater = Updater(opt=opt, evaluator=loss_fcn, model_init=model.init)
    
    rng, subkey = random.split(rng)
        
    dummy_input,_,_,_ = process_batch(jax.random.PRNGKey(0), train_inputs[:args.bs], 0,mask_prob=args.mask_prob, chunk_size=args.chunk_size, 
                                    mask_individual=args.mask_single, mask_indicators=args.mask_indicators)
    state = updater.init_params(subkey, x=utils.tree_stack(dummy_input)) #

    print("Number of parameters:", utils.count_params(state.params) / 1e6, "Million")
    
    # logger
    logger = Logger(name = args.wandb_log_name,
                    config={
                    "seed":args.seed,
                    "exp": args.exp,
                    "dataset": "all",
                    "lr": args.lr,
                    "weight_decay": args.wd,
                    "batchsize": args.bs,
                    "num_epochs": args.epochs,
                    "dropout": args.dropout,
                    "model_size":args.model_size,
                    "chunk_size":args.chunk_size,
                    "num_layers":args.num_layers,
                    "augment":args.augment},
                    log_wandb = args.use_wandb,
                    save_checkpoints=True,
                    save_interval=10,
                    log_interval=args.log_interval,
                    checkpoint_dir=os.path.join('checkpoints',args.exp,str(time.time())))
    logger.init(is_save_config=True)

    # Training loop
    for epoch in range(args.epochs):
        rng,subkey = random.split(rng)
        
        if args.augment:
            images,_ = augment_whole(subkey,train_inputs,np.zeros(len(train_inputs)),num_p=args.num_augment,keep_original=False)
        else:
            images = train_inputs
        rng, subkey = random.split(rng)
        images, _ = shuffle_data(subkey, images, np.zeros(len(images)),chunks=args.num_checkpoints*args.bs)
        rng, subkey = random.split(rng)
        masked_ins, masked_labels, positions, non_masked_positions = process_batch(subkey, images, MASK_TOKEN, 
                                                      mask_prob=args.mask_prob,
                                                      chunk_size=args.chunk_size,
                                                      mask_individual=args.mask_single, 
                                                      mask_indicators=args.mask_indicators)
        batches = data_iterator(masked_ins, masked_labels, positions,non_masked_positions, batchsize=args.bs, skip_last=True)

        train_all_loss = []
        for it, batch in enumerate(batches):
            #batch["input"] = utils.tree_stack(batch["input"])
            state, train_metrics = updater.train_step(state, (batch['masked_input'],batch['input'],batch['position'],batch['non_position'],batch['variances']))
            #logger.log(state, train_metrics)
            train_all_loss.append(train_metrics['train/loss'].item())
        train_metrics = {'train/loss':np.mean(train_all_loss)}
            
        # Validate every epoch
        rng, subkey = random.split(rng)
        masked_ins, masked_labels, positions,non_masked_positions = process_batch(subkey, val_inputs, MASK_TOKEN, 
                                                      mask_prob=args.mask_prob,
                                                      chunk_size=args.chunk_size,
                                                      mask_individual=args.mask_single, 
                                                      mask_indicators=args.mask_indicators)
        batches = data_iterator(masked_ins, masked_labels, positions,non_masked_positions, batchsize=args.bs, skip_last=True)
        val_all_loss = []
        for it, batch in enumerate(batches):
            #batch["input"] = utils.tree_stack(batch["input"])
            state, val_metrics = updater.val_step(state, (batch['masked_input'],batch['input'],batch['position'],batch['non_position'],batch['variances']))
            #logger.log(state, val_metrics)
            val_all_loss.append(val_metrics['val/loss'].item())
        val_metrics = {'val/loss':np.mean(val_all_loss)}
            
        logger.log(state, train_metrics, val_metrics)