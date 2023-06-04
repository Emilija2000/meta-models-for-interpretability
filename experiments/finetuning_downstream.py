import argparse
import jax
from jax import random
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import optax
import os
from typing import Tuple, List, Iterator
import time

from augmentations import augment_batch
from meta_transformer import utils, preprocessing
from meta_transformer.meta_model import MetaModelConfig as ModelConfig
from model_zoo_jax import load_nets, shuffle_data, CrossEntropyLoss, MSELoss, Updater

from finetuning import load_pretrained_state, get_meta_model_fcn
from pretraining import Logger, process_batch

def split_data(data: list, labels: list):
    split_index = int(len(data)*0.8)
    return (data[:split_index], labels[:split_index], 
            data[split_index:], labels[split_index:])

def flatten(x):
    return jax.flatten_util.ravel_pytree(x)[0]

def is_fine(params: dict):
    """Return false if std or mean is too high."""
    flat = flatten(params)
    if flat.std() > 5.0 or jnp.abs(flat.mean()) > 5.0:
        return False
    else:
        return True

def filter_data(data: List[dict], labels: List[ArrayLike]):
    """Given a list of net params, filter out those
    with very large means or stds."""
    assert len(data) == len(labels)
    f_data, f_labels = zip(*[(x, y) for x, y in zip(data, labels) if is_fine(x)])
    print(f"Filtered out {len(data) - len(f_data)} nets.\
          That's {100*(len(data) - len(f_data))/len(data):.2f}%.")
    return np.array(f_data), np.array(f_labels)


def data_iterator(inputs: jnp.ndarray, labels: jnp.ndarray, batchsize: int = 1048, skip_last: bool = False) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Iterate over the data in batches."""
    for i in range(0, len(inputs), batchsize):
        if skip_last and i + batchsize > len(inputs):
            break
        yield dict(input=inputs[i:i + batchsize], 
                   label=labels[i:i + batchsize])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training run')
    # training parameters
    parser.add_argument('--lr', type=float, help='Learning rate', default=5e-5)
    parser.add_argument('--wd', type=float, help='Weight decay', default=1e-3)
    parser.add_argument('--dropout',type=float,help='Meta-transformer dropout', default=0.1)
    parser.add_argument('--bs', type=int, help='Batch size', default=32)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=25)
    # meta-model
    parser.add_argument('--model_size',type=int,help='MetaModelClassifier model_size parameter',default=4*32)
    parser.add_argument('--num_layers',type=int,help='num of transformer layers',default=12)
    parser.add_argument('--chunk_size',type=int,help='meta model chunk size',default=128)
    parser.add_argument('--num_classes',type=int,help='Number of classes for this downstream task',default=10)
    # pretrained meta-model
    parser.add_argument('--model_type',type=str, help="Options: classifier, plus_linear, replaced_last", default='classifier')
    parser.add_argument('--pretrained_path',type=str,help='Pretrained model weights',default=None)
    parser.add_argument('--mask_single',action='store_true',help='Mask each weight individually')
    parser.add_argument('--mask_indicators',action='store_true',help='Include binary mask indicators to meta-model chunked input')
    # data
    parser.add_argument('--task', type=str, help='Task to train on.', default="class_dropped")
    parser.add_argument('--data_dir',type=str,default='/rds/user/ed614/hpc-work/model_zoo_datasets/downstream_droppedcls_mnist_smallCNN_fixed_zoo')
    parser.add_argument('--num_checkpoints',type=int,default=1)
    parser.add_argument('--num_networks',type=int,default=None)
    # augmentations
    parser.add_argument('--augment', action='store_true', help='Use permutation augmentation')
    parser.add_argument('--num_augment',type=int,default=3)
    #logging
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb')
    parser.add_argument('--wandb_log_name', type=str, default="meta-transformer-fine-tuning-fixed-mnist")
    parser.add_argument('--log_interval',type=int, default=50)
    parser.add_argument('--seed',type=int, help='PRNG key seed',default=42)
    parser.add_argument('--exp', type=str, default="baseline")
    args = parser.parse_args()
    
    rng = random.PRNGKey(args.seed)

    # Load model zoo checkpoints
    print(f"Loading model zoo: {args.data_dir}")
    inputs, all_labels = load_nets(n=args.num_networks, 
                                   data_dir=args.data_dir,
                                   flatten=False,
                                   num_checkpoints=args.num_checkpoints)
    
    print(f"Training task: {args.task}.")
    labels = all_labels[args.task]
    
    # Filter (high variance)
    filtered_inputs, filtered_labels = filter_data(inputs, labels)
    
    #unpreprocess = preprocessing.get_unpreprocess(filtered_inputs[0], args.chunk_size)
    
    # Shuffle checkpoints before splitting
    rng, subkey = random.split(rng)
    filtered_inputs, filtered_labels = shuffle_data(subkey,filtered_inputs,filtered_labels,chunks=args.num_checkpoints)
    
    train_inputs, train_labels, val_inputs, val_labels = split_data(filtered_inputs, filtered_labels)
    val_data = {"input": utils.tree_stack(val_inputs), "label": val_labels}

    steps_per_epoch = len(train_inputs) // args.bs
    print()
    print(f"Number of training examples: {len(train_inputs)}.")
    print("Steps per epoch:", steps_per_epoch)
    print("Total number of steps:", steps_per_epoch * args.epochs)
    print()
    
    model_config = ModelConfig(
        model_size=args.model_size,
        num_heads=8,
        num_layers=args.num_layers,
        dropout_rate=args.dropout,
        use_embedding=True,
    )
    model = get_meta_model_fcn(model_config, args.num_classes, args.model_type)

    # Initialization
    if args.num_classes==1:
        evaluator = MSELoss(model.apply)
    else:
        evaluator = CrossEntropyLoss(model.apply, args.num_classes)
    
    opt = optax.adamw(learning_rate=args.lr, weight_decay=args.wd)
    updater = Updater(opt=opt, evaluator=evaluator, model_init=model.init)
    
    rng, subkey = random.split(rng)
    dummy_input,_,_ = process_batch(jax.random.PRNGKey(0), train_inputs[:args.bs], 0,
                                    mask_prob=0, chunk_size=args.chunk_size, 
                                    mask_individual=args.mask_single, 
                                    mask_indicators=args.mask_indicators)
    state = updater.init_params(subkey, x=utils.tree_stack(dummy_input)) #
    
    # switch params to pretrained
    if args.pretrained_path is not None:
        state = load_pretrained_state(state, args.pretrained_path)

    print("Number of parameters:", utils.count_params(state.params) / 1e6, "Million")
    
    # logger
    logger = Logger(name = args.wandb_log_name,
                    config={
                        "exp": args.exp,
                        "dataset": os.path.basename(args.data_dir),
                        "lr": args.lr,
                        "weight_decay": args.wd,
                        "batchsize": args.bs,
                        "num_epochs": args.epochs,
                        "target_task": args.task,
                        "dropout": args.dropout,
                        "model_type": args.model_type,
                        "model_size":args.model_size,
                        "num_layers":args.num_layers,
                        "augment":args.augment},
                    log_wandb = args.use_wandb,
                    save_checkpoints=True,
                    log_interval=args.log_interval,
                    save_interval=50,
                    checkpoint_dir=os.path.join('checkpoints',args.exp,str(time.time())))
    logger.init(is_save_config=True)

    # Training loop
    for epoch in range(args.epochs):
        rng,subkey = random.split(rng)
        if args.augment:
            images,labels = augment_batch(subkey,train_inputs,train_labels,num_p=args.num_augment,keep_original=False)
        else:
            images,labels = train_inputs,train_labels
        rng, subkey = random.split(rng)
        images, labels = shuffle_data(subkey, images, labels)
        images, _, _ = process_batch(subkey, images, 0, 
                                            mask_prob=0,
                                            chunk_size=args.chunk_size,
                                            mask_individual=args.mask_single, 
                                            mask_indicators=args.mask_indicators)
        batches = data_iterator(images, labels, batchsize=args.bs, skip_last=True)

        train_all_acc = []
        train_all_loss = []
        for it, batch in enumerate(batches):
            batch["input"] = utils.tree_stack(batch["input"])
            state, train_metrics = updater.train_step(state, (batch['input'],batch['label']))
            logger.log(state, train_metrics)
            train_all_acc.append(train_metrics['train/acc'].item())
            train_all_loss.append(train_metrics['train/loss'].item())
        train_metrics = {'train/acc':np.mean(train_all_acc), 'train/loss':np.mean(train_all_loss)}
            
        # Validate every epoch
        images, labels = shuffle_data(subkey, val_inputs, val_labels)
        images, _, _ = process_batch(subkey, images, 0, 
                                            mask_prob=0,
                                            chunk_size=args.chunk_size,
                                            mask_individual=args.mask_single, 
                                            mask_indicators=args.mask_indicators)
        batches = data_iterator(images, labels, batchsize=32, skip_last=True)
        val_all_acc = []
        val_all_loss = []
        for it, batch in enumerate(batches):
            batch["input"] = utils.tree_stack(batch["input"])
            state, val_metrics = updater.val_step(state, (batch['input'],batch['label']))
            #logger.log(state, val_metrics)
            val_all_acc.append(val_metrics['val/acc'].item())
            val_all_loss.append(val_metrics['val/loss'].item())
        val_metrics = {'val/acc':np.mean(val_all_acc), 'val/loss':np.mean(val_all_loss)}
            
        logger.log(state, train_metrics, val_metrics)