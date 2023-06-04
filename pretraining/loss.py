from dataclasses import dataclass
from typing import Callable
import functools
import jax
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

@dataclass(frozen=True)
class MWMLossMSE(LossFcn):
    non_masked: bool = True
    
    def masked_loss_fn(self,predictions, targets, positions):
        #diff=[]
        #jax.debug.print("first prediction{}",jnp.isnan(predictions[0]))
        #jax.debug.print("how many masked: {}",jnp.sum(positions[0]))
        #for i,pos in enumerate(positions):
        #    #n_masked = jnp.min(jnp.array([jnp.sum(pos),1]).astype(jnp.int32))
        #    #print(n_masked)
        #    sd = jnp.square(predictions[i]-targets[i])
        #    masked_sd = jnp.multiply(pos, sd)
        #    #diff.append(jnp.sum(masked_sd)/n_masked)
        #    diff.append(jnp.sum(masked_sd))
        #    #if jnp.isnan(diff[-1]):
        #    #    print('NOT A NUMBER')
        #    #    exit(-1)
        #jax.debug.print("diff: {}",diff)
        #return jnp.mean(jnp.array(diff))

        diff = jnp.square(predictions-targets)
        diff = jnp.multiply(positions, diff)
        res = jnp.sum(diff)/(jnp.sum(positions)+0.0001)
        return res
    
    def loss_fn(self, logits, targets, masked=None):
        masked_loss = self.masked_loss_fn(logits,targets,masked) if masked is not None else 0
        loss = masked_loss
        if self.non_masked:
            unmasked_loss = self.masked_loss_fn(logits,targets,[1-m for m in masked])
            loss = loss + unmasked_loss
        return loss
 
@dataclass(frozen=True)  
class MWMLossCosine(LossFcn):
    non_masked: bool = True
    
    def masked_loss_fn(self,predictions, targets, positions):
        diff = []
        for i,pos in enumerate(positions):
            a = jnp.ravel(jnp.multiply(pos, predictions[i]))
            b = jnp.ravel(jnp.multiply(pos, targets[i]))
            diff.append(self.cosine_distance(a,b))
        return jnp.mean(jnp.array(diff))

    def cosine_distance(y_true, y_pred):
        y_true_norm = jnp.linalg.norm(y_true, axis=-1, keepdims=True)
        y_pred_norm = jnp.linalg.norm(y_pred, axis=-1, keepdims=True)
        dot_product = jnp.sum(y_true * y_pred, axis=-1, keepdims=True)
        cosine_distance = jnp.mean(1.0 - (dot_product / (y_true_norm * y_pred_norm)))
        return cosine_distance
    
    def loss_fn(self, logits, targets, masked=None):
        masked_loss = self.masked_loss_fn(logits,targets,masked) if masked is not None else 0
        loss = masked_loss
        if self.non_masked:
            unmasked_loss = self.masked_loss_fn(logits,targets,1-masked)
            loss = loss + unmasked_loss
        return loss
    