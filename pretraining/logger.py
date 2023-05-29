import chex
import jax
import numpy as np
import os
import pickle
import json
from typing import Optional

from pretraining.train import TrainState
import wandb

def model_save(ckpt_dir: str, state) -> None:
    "credit: https://github.com/deepmind/dm-haiku/issues/18?fbclid=IwAR0aSk2OgYCIn3YKFrDoEnSYU1xRYzywuypVQlunsZHn2w5y1vpN9_b8QXM"
    with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
        for x in jax.tree_util.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)

def model_restore(ckpt_dir):
    "credit: https://github.com/deepmind/dm-haiku/issues/18?fbclid=IwAR0aSk2OgYCIn3YKFrDoEnSYU1xRYzywuypVQlunsZHn2w5y1vpN9_b8QXM"
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)
 
    leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_util.tree_unflatten(treedef, flat_state)


@chex.dataclass
class Logger:
    name: str
    config: dict
    log_wandb:Optional[bool] = True
    log_interval: Optional[int] = 50
    save_checkpoints: Optional[bool] = True
    checkpoint_dir: Optional[str] = "checkpoints"
    save_interval: Optional[int] = 20
    
    def init(self,is_save_config=True):
        if self.log_wandb:
            wandb.init(config=self.config, project=self.name)
        if is_save_config:
            self.save_config()
        self.savestep=0

    def wandb_log(self,
            state: TrainState,
            train_metrics: dict,
            val_metrics: dict = None):
        metrics = train_metrics
        if val_metrics is not None:
            metrics.update(val_metrics)
        metrics = {k: float(v) for k, v in metrics.items() if k != "step"}
        wandb.log(metrics, step=state.step)
        print(", ".join([f"{k}: {round(v, 3)}" for k, v in metrics.items()]))
            
    def save_checkpoint(self, state:TrainState, train_metrics:dict, val_metrics:dict=None):
        checkpoint_name = os.path.join(self.checkpoint_dir,str(state.step))
        if not os.path.exists(checkpoint_name):
            os.makedirs(checkpoint_name)
        model_save(checkpoint_name, state.params)
        
        metrics = train_metrics
        metrics.update(val_metrics)
        
        metrics = {key: value.item() for key,value in metrics.items()}
        with open(os.path.join(checkpoint_name, "metrics.json"), "w") as f:
            json.dump(dict(metrics), f, indent=4)
            
    def save_config(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        with open(os.path.join(self.checkpoint_dir, "config.json"), "w") as f:
            config = dict(self.config)
            json.dump(dict(config), f, indent=4)
        
    def log(self, 
            state: TrainState,
            train_metrics: dict,
            val_metrics: dict = None,
            last=False):
        if self.log_wandb and (state.step % self.log_interval == 0 or val_metrics is not None):
            self.wandb_log(state,train_metrics,val_metrics)
        if val_metrics is not None and self.save_checkpoints: 
            self.savestep = self.savestep+1
            if (self.savestep % self.save_interval == 0) or last:
                self.save_checkpoint(state,train_metrics,val_metrics)
            
        
        