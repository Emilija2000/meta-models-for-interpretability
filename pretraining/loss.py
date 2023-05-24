from dataclasses import dataclass
from typing import Callable
import functools
import jax.numpy as jnp
from jax import jit

@dataclass(frozen=True)
class LossFcn:
    model_apply: Callable
    
    def loss_fn(self,logits,targets,masked=None):
        pass
    
    @functools.partial(jit, static_argnums=(0))
    def train_metrics(self, params, rng, data,state=None):
        inputs = data[0]
        targets = data[1]
        mask_positions = data[2]
        if state is None:
            logits = self.model_apply(params, rng, inputs, True) 
        else:
            logits,state=self.model_apply(params,state,rng,inputs,True)
        loss = self.loss_fn(logits,targets,mask_positions)
        return loss, state
    
    @functools.partial(jit, static_argnums=(0))
    def val_metrics(self, params, rng, data,state=None):
        inputs = data[0]
        targets = data[1]
        mask_positions = data[2]
        if state is None:
            logits = self.model_apply(params, rng, inputs, False)
        else:
            logits,state = self.model_apply(params, state, rng, inputs, False)
        loss = self.loss_fn(logits,targets,mask_positions)
        return loss, state
    
class MWMLossMSE(LossFcn):
    
    def masked_loss_fn(self,predictions, targets, positions):
        diff = []
        for i,pos in enumerate(positions):
            diff.append(jnp.square(jnp.matmul(jnp.transpose(pos),(predictions[i]-targets[i]))))
        return jnp.mean(jnp.array(diff))
    
    def loss_fn(self, logits, targets, masked=None):
        masked_loss = self.masked_loss_fn(logits,targets,masked) if masked is not None else 0
        unmasked_loss = self.masked_loss_fn(logits,targets,[1-m for m in masked])
        loss = masked_loss + unmasked_loss
        return loss
    