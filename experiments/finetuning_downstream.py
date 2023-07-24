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

# model
from model.meta_model import MetaModelConfig as ModelConfig
from finetuning import load_pretrained_state, get_meta_model_fcn

# data
from model_zoo_jax import load_data, shuffle_data, CrossEntropyLoss, MSELoss, Updater
from torchload import load_modelzoo
from augmentations import augment_batch

from pretraining import Logger, process_batch

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
    parser.add_argument('--num_heads',type=int,help='num of MHA heads',default=8)
    parser.add_argument('--chunk_size',type=int,help='meta model chunk size',default=128)
    parser.add_argument('--num_classes',type=int,help='Number of classes for this downstream task',default=10)
    # pretrained meta-model
    parser.add_argument('--model_type',type=str, help="Options: classifier, plus_linear, replace_last", default='classifier')
    parser.add_argument('--pretrained_path',type=str,help='Pretrained model weights',default=None)
    parser.add_argument('--mask_single',action='store_true',help='Mask each weight individually')
    parser.add_argument('--mask_indicators',action='store_true',help='Include binary mask indicators to meta-model chunked input')
    # data
    parser.add_argument('--dataset_type',type=str,help='My dataset or external torch dataset. Values:myzoo or torchzoo',default='myzoo')
    parser.add_argument('--filter', action='store_true', help='Filter large variance in model zoo')
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

    # Load data
    if args.dataset_type == 'myzoo':
        rng,subkey = random.split()
        train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels = load_data(subkey, args.data_dir, args.task,args.num_networks,args.num_checkpoints, is_filter=args.filter)
    else:
        train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels = load_modelzoo(args.data_dir, args.task, epochs=list(range(0,51,50//args.num_checkpoints))[1:])

        # Keep a subset for finetuning/baseline
        splitkey = random.PRNGKey(123)
        train_inputs, train_labels = shuffle_data(splitkey, train_inputs, train_labels)
        train_inputs, train_labels = train_inputs[-500:], train_labels[-500:]
        val_inputs, val_labels = shuffle_data(splitkey, val_inputs, val_labels)
        val_inputs, val_labels =  val_inputs[-400:], val_labels[-400:]
        test_inputs, test_labels = shuffle_data(splitkey, test_inputs, test_labels)
        test_inputs, test_labels = test_inputs[-400:], test_labels[-400:]

    steps_per_epoch = len(train_inputs) // args.bs
    print()
    print(f"Number of training examples: {len(train_inputs)}, {len(train_labels)}.")
    print(f"Number of val examples: {len(val_inputs)}, {len(val_labels)}")
    print(f"Number of test examples: {len(test_inputs)}, {len(test_labels)}")
    print()
    print("Steps per epoch:", steps_per_epoch)
    print("Total number of steps:", steps_per_epoch * args.epochs)
    print()
    
    model_config = ModelConfig(
        model_size=args.model_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout,
        use_embedding=True,
        max_seq_len=None
    )
    model = get_meta_model_fcn(model_config, args.num_classes, args.model_type)

    # Initialization
    if args.num_classes==1:
        evaluator = MSELoss(model.apply)
    else:
        evaluator = CrossEntropyLoss(model.apply, args.num_classes)
    
    opt = optax.adamw(learning_rate=args.lr, weight_decay=args.wd/args.lr)
    updater = Updater(opt=opt, evaluator=evaluator, model_init=model.init)
    
    rng, subkey = random.split(rng)
    dummy_input,_,_,_ = process_batch(jax.random.PRNGKey(0), train_inputs[:args.bs], 0,
                                    mask_prob=0, chunk_size=args.chunk_size, 
                                    mask_individual=args.mask_single, 
                                    mask_indicators=args.mask_indicators,
                                    resample_zeromasks=False)
    state = updater.init_params(subkey, x=dummy_input) #
    
    # switch params to pretrained
    if args.pretrained_path is not None:
        state = load_pretrained_state(state, args.pretrained_path)
        checkpoints_dir = checkpoint_dir=os.path.join('checkpoints','finetuning',args.exp,str(time.time()))
    else:
        checkpoints_dir=os.path.join('checkpoints','baselines',args.exp,str(time.time()))

    print("Number of parameters:", sum(x.size for x in jax.tree_util.tree_leaves(state.params)) / 1e6, "Million")
    
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
                    save_interval=100,
                    checkpoint_dir=checkpoints_dir)
    logger.init(is_save_config=True)

    best = None
    best_loss = 1000000.0
    
    # Training loop
    for epoch in range(args.epochs):
        rng,subkey = random.split(rng)
        if args.augment:
            images,labels = augment_batch(subkey,train_inputs,train_labels,num_p=args.num_augment,keep_original=False)
        else:
            images,labels = train_inputs,train_labels
        rng, subkey = random.split(rng)
        images, labels = shuffle_data(subkey, images, labels)
        images, _, _,_ = process_batch(subkey, images, 0, 
                                            mask_prob=0,
                                            chunk_size=args.chunk_size,
                                            mask_individual=args.mask_single, 
                                            mask_indicators=args.mask_indicators,
                                            resample_zeromasks=False)
        batches = data_iterator(images, labels, batchsize=args.bs, skip_last=True)

        train_all_acc = []
        train_all_loss = []
        for it, batch in enumerate(batches):
            state, train_metrics = updater.train_step(state, (batch['input'],batch['label']))
            logger.log(state, train_metrics)
            train_all_acc.append(train_metrics['train/acc'].item())
            train_all_loss.append(train_metrics['train/loss'].item())
        
        if args.num_classes==1:
            ss_res = jnp.sum(jnp.array(train_all_acc))
            ss_tot = jnp.sum(jnp.square(labels - jnp.mean(labels)))
            train_metrics = {'train/r_squared': 1.0 - (ss_res / ss_tot),'train/diff':np.mean(train_all_acc), 'train/loss':np.mean(train_all_loss)}
        else:        
            train_metrics = {'train/acc':np.mean(train_all_acc), 'train/loss':np.mean(train_all_loss)}
            
        # Validate every epoch
        images, _, _,_ = process_batch(subkey, val_inputs, 0, 
                                            mask_prob=0,
                                            chunk_size=args.chunk_size,
                                            mask_individual=args.mask_single, 
                                            mask_indicators=args.mask_indicators,
                                            resample_zeromasks=False)
        batches = data_iterator(images, val_labels, batchsize=args.bs, skip_last=True)
        val_all_acc = []
        val_all_loss = []
        for it, batch in enumerate(batches):
            state, val_metrics = updater.val_step(state, (batch['input'],batch['label']))
            val_all_acc.append(val_metrics['val/acc'].item())
            val_all_loss.append(val_metrics['val/loss'].item())
            
        if args.num_classes==1:
            ss_res = jnp.sum(jnp.array(val_all_acc))
            ss_tot = jnp.sum(jnp.square(val_labels - jnp.mean(val_labels)))
            val_metrics = {'val/r_squared': 1.0 - (ss_res / ss_tot),'val/diff':np.mean(val_all_acc), 'val/loss':np.mean(val_all_loss)}
        else:    
            val_metrics = {'val/acc':np.mean(val_all_acc), 'val/loss':np.mean(val_all_loss)}
            
        if val_metrics['val/loss']<best_loss:
            best_loss = val_metrics['val/loss']
            best = state
            
        logger.log(state, train_metrics, val_metrics)
        
    ## TEST
    images, _, _,_ = process_batch(subkey, test_inputs, 0, 
                                        mask_prob=0,
                                        chunk_size=args.chunk_size,
                                        mask_individual=args.mask_single, 
                                        mask_indicators=args.mask_indicators,
                                        resample_zeromasks=False)
    batches = data_iterator(images, test_labels, batchsize=args.bs, skip_last=True)
    test_all_acc = []
    test_all_loss = []
    for it, batch in enumerate(batches):
        state, test_metrics = updater.val_step(best, (batch['input'],batch['label']))
        test_all_acc.append(test_metrics['val/acc'].item())
        test_all_loss.append(test_metrics['val/loss'].item())
        
    if args.num_classes==1:
        ss_res = jnp.sum(jnp.array(test_all_acc))
        ss_tot = jnp.sum(jnp.square(test_labels - jnp.mean(test_labels)))
        test_metrics = {'test/r_squared': 1.0 - (ss_res / ss_tot),'test/diff':np.mean(test_all_acc), 'test/loss':np.mean(test_all_loss)}
    else:    
        test_metrics = {'test/acc':np.mean(test_all_acc), 'test/loss':np.mean(test_all_loss)}
    print("Test results")
    logger.log(state, test_metrics)