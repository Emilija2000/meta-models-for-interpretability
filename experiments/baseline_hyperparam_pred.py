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
from model.meta_model import MetaModelConfig as ModelConfig
from model_zoo_jax import shuffle_data, CrossEntropyLoss, MSELoss, MSLELoss, Updater

from finetuning import load_pretrained_state, get_meta_model_fcn
from pretraining import Logger, process_batch, TrainState

from torchload import load_dataset
from chunking import preprocessing

def split_data(data: list, labels: list, chunks:int=1):
    split_index = int(len(data)*0.7)
    split_index -= split_index % chunks
    split_index_1 = int(len(data)*0.85)
    split_index_1 -= split_index_1 % chunks
    return (data[:split_index], labels[:split_index], 
            data[split_index:split_index_1], labels[split_index:split_index_1],
            data[split_index_1:], labels[split_index_1:])

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
    parser.add_argument('--num_heads',type=int,help='num of MHA heads',default=8)
    parser.add_argument('--chunk_size',type=int,help='meta model chunk size',default=128)
    parser.add_argument('--num_classes',type=int,help='Number of classes for this downstream task',default=10)
    # pretrained meta-model
    parser.add_argument('--model_type',type=str, help="Options: classifier, plus_linear, replaced_last", default='classifier')
    parser.add_argument('--pretrained_path',type=str,help='Pretrained model weights',default=None)
    parser.add_argument('--mask_single',action='store_true',help='Mask each weight individually')
    parser.add_argument('--mask_indicators',action='store_true',help='Include binary mask indicators to meta-model chunked input')
    # data
    parser.add_argument('--task', type=str, help='Task to train on.', default="train_accuracy")
    parser.add_argument('--data_dir',type=str,default='/rds/user/ed614/hpc-work/model_zoo_datasets/hyp/mnist')
    # augmentations
    parser.add_argument('--augment', action='store_true', help='Use permutation augmentation')
    parser.add_argument('--num_augment',type=int,default=3)
    #logging
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb')
    parser.add_argument('--wandb_log_name', type=str, default="meta-transformer-hyperparam-prediction")
    parser.add_argument('--log_interval',type=int, default=50)
    parser.add_argument('--seed',type=int, help='PRNG key seed',default=42)
    parser.add_argument('--exp', type=str, default="baseline")
    args = parser.parse_args()
    
    rng = random.PRNGKey(args.seed)
    #LOG_TASKS = ['config.learning_rate']
    LOG_TASKS=[]
    DISCRETE_TASKS = ['config.activation','config.w_init','config.optimizer']

    # Load model zoo checkpoints
    print(f"Loading model zoo: {args.data_dir}")
    inputs, all_labels = load_dataset(args.data_dir)
    
    print(f"Training task: {args.task}.")
    labels = all_labels[args.task]
    if args.task in DISCRETE_TASKS:
        print(np.unique(labels))
        args.num_classes = int(len(jnp.unique(labels).tolist()))
    if args.num_classes==1:
        labels = labels.reshape(-1,1)
    
    # Shuffle checkpoints before splitting
    rng, subkey = random.split(rng)
    inputs, labels = shuffle_data(subkey,inputs,labels,chunks=9)#9 from the same training run
    print('splitting...')
    train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels = split_data(inputs, labels,chunks=9)
    
    steps_per_epoch = len(train_inputs) // args.bs
    print()
    print(f"Number of training examples: {len(train_inputs)}.")
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
        if args.task in LOG_TASKS:
            evaluator = MSLELoss(model.apply)
        else:
            evaluator = MSELoss(model.apply)
    else:
        evaluator = CrossEntropyLoss(model.apply, args.num_classes)
    
    opt = optax.adamw(learning_rate=args.lr, weight_decay=args.wd/args.lr)
    updater = Updater(opt=opt, evaluator=evaluator, model_init=model.init)
    
    dummy_input = preprocessing.batch_process_layerwise(train_inputs[:args.bs],args.chunk_size)
    rng,subkey=random.split(rng)
    state = updater.init_params(subkey, x=dummy_input) #
    
    # switch params to pretrained
    if args.pretrained_path is not None:
        state = load_pretrained_state(state, args.pretrained_path)

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
                    save_interval=args.epochs//2,
                    checkpoint_dir=os.path.join('checkpoints',args.exp,str(time.time())))
    logger.init(is_save_config=True)
    
    print('chunking val')
    val_inputs = preprocessing.batch_process_layerwise(val_inputs,args.chunk_size)
    
    best_val = 100000
    best_model = None
    
    # Training loop
    for epoch in range(args.epochs):
        print('augment and shuffle')
        rng,subkey = random.split(rng)
        if args.augment:
            images,labels = augment_batch(subkey,train_inputs,train_labels,num_p=args.num_augment,keep_original=False)
        else:
            images,labels = train_inputs,train_labels
        
        rng, subkey = random.split(rng)
        images, labels = shuffle_data(subkey, images, labels)
        print('chunking train')
        images = preprocessing.batch_process_layerwise(images,args.chunk_size)
        
        batches = data_iterator(images, labels, batchsize=args.bs, skip_last=True)
        print('training loop')
        train_all_acc = []
        train_all_loss = []
        for it, batch in enumerate(batches):
            state, train_metrics = updater.train_step(state, (batch['input'],batch['label']))
            logger.log(state, train_metrics)
            train_all_acc.append(train_metrics['train/acc'].item())
            train_all_loss.append(train_metrics['train/loss'].item())
        
        ss_res = jnp.sum(jnp.array(train_all_acc))
        ss_tot = jnp.sum(jnp.square(labels - jnp.mean(labels)))
        train_metrics = {'train/r_squared': 1.0 - (ss_res / ss_tot),'train/acc':np.mean(train_all_acc), 'train/loss':np.mean(train_all_loss)}
            
        # Validate every epoch

        batches = data_iterator(val_inputs, val_labels, batchsize=args.bs, skip_last=True)
        val_all_acc = []
        val_all_loss = []
        for it, batch in enumerate(batches):
            state, val_metrics = updater.val_step(state, (batch['input'],batch['label']))
            #logger.log(state, val_metrics)
            val_all_acc.append(val_metrics['val/acc'].item())
            val_all_loss.append(val_metrics['val/loss'].item())
        
        ss_res = jnp.sum(jnp.array(val_all_acc))
        ss_tot = jnp.sum(jnp.square(val_labels - jnp.mean(val_labels)))
        val_metrics = {'val/r_squared': 1.0 - (ss_res / ss_tot),'val/acc':np.mean(val_all_acc), 'val/loss':np.mean(val_all_loss)}
            
        logger.log(state, train_metrics, val_metrics)
        
        if best_val > val_metrics['val/loss']:
            best_val = val_metrics['val/loss']
            best_model = state.params.copy()
            
    logger.save_checkpoint(TrainState(step=state.step,rng=state.rng,opt_state=state.opt_state,params=best_model,model_state=state.model_state),
                           train_metrics, val_metrics)
        
    batches = data_iterator(test_inputs, test_labels, batchsize=args.bs, skip_last=True)
    val_all_acc = []
    val_all_loss = []
    for it, batch in enumerate(batches):
        state, val_metrics = updater.val_step(state, (batch['input'],batch['label']))
        #logger.log(state, val_metrics)
        val_all_acc.append(val_metrics['val/acc'].item())
        val_all_loss.append(val_metrics['val/loss'].item())
    
    ss_res = jnp.sum(jnp.array(val_all_acc))
    ss_tot = jnp.sum(jnp.square(val_labels - jnp.mean(val_labels)))
    val_metrics = {'test/r_squared': 1.0 - (ss_res / ss_tot),'test/acc':np.mean(val_all_acc), 'test/loss':np.mean(val_all_loss)}
        
    logger.log(state, train_metrics, val_metrics)
    print(val_metrics)