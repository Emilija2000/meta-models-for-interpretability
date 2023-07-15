from dataclasses import dataclass
from typing import Callable
import functools
import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax import jit


@dataclass(frozen=True)
class LossFcn:
    model_apply: Callable
    
    def loss_fn(self,logits,targets,masked=None,non_masked=None):
        pass
    
    @functools.partial(jit, static_argnums=(0))
    def train_metrics(self, params, rng, data,state=None):
        inputs = data[0]
        #jax.debug.print('inputs: {}', inputs)
        targets = data[1]
        #jax.debug.print('targets: {}',targets)
        mask_positions = data[2]
        non_mask_positions = data[3]
        variances=data[4]
        if state is None:
            logits = self.model_apply(params, rng, inputs, True) 
        else:
            logits,state=self.model_apply(params,state,rng,inputs,True)
        #jax.debug.print('predictions: {}',logits)
        
        loss = self.loss_fn(logits,targets,mask_positions,non_mask_positions,variances)
        return loss, state
    
    @functools.partial(jit, static_argnums=(0))
    def val_metrics(self, params, rng, data,state=None):
        inputs = data[0]
        targets = data[1]
        mask_positions = data[2]
        non_mask_positions=data[3]
        variances=data[4]
        if state is None:
            logits = self.model_apply(params, rng, inputs, False)
        else:
            logits,state = self.model_apply(params, state, rng, inputs, False)
        loss = self.loss_fn(logits,targets,mask_positions,non_mask_positions,variances)
        return loss, state

@dataclass(frozen=True)
class MWMLossMSE(LossFcn):
    non_masked: bool = True
    
    @functools.partial(jit, static_argnums=(0))
    def masked_loss_fn(self,predictions, targets, positions):
        diff = jnp.square(predictions-targets)
        diff = jnp.multiply(positions, diff) #shape [batch_size, num_chunks, chunk_size]
        res = jnp.sum(diff)/(jnp.sum(positions)+0.0001)
        return res
    
    @functools.partial(jit, static_argnums=(0))
    def loss_fn(self, logits, targets, masked=None, non_masked=None,variances=None,alpha=1.0):
        masked_loss = self.masked_loss_fn(logits,targets,masked) if masked is not None else 0
        loss = masked_loss
        if self.non_masked:
            unmasked_loss = self.masked_loss_fn(logits,targets,non_masked) if non_masked is not None else 0
            loss = loss + alpha*unmasked_loss
        return loss
    
    @functools.partial(jit, static_argnums=(0))
    def r2_score(self, predictions, targets, positions):
        # Mask the targets and predictions
        masked_targets = jnp.multiply(targets, positions)
        masked_predictions = jnp.multiply(predictions, positions)

        # Compute the total and residual sum of squares
        tss = jnp.sum(jnp.square(masked_targets - jnp.sum(masked_targets)/jnp.sum(positions)))
        rss = jnp.sum(jnp.square(masked_targets - masked_predictions))

        # Compute the R2 score
        r2 = 1 - (rss / (tss + 0.0001))  # added small constant for numerical stability

        return r2

    
@dataclass(frozen=True)
class MWMLossMseNormalized(LossFcn):
    non_masked: bool = True
    
    @functools.partial(jit, static_argnums=(0))
    def masked_loss_fn(self,predictions, targets, positions,variances):
        diff = jnp.square(predictions-targets)
        diff = jnp.multiply(positions, diff) #shape [batch_size, num_chunks, chunk_size]
        #jax.debug.print('diff: {}',diff)
        diff = jnp.sum(diff, axis=(0,2)) #shape [num_chunks,]
        diff = jnp.reshape(diff, (diff.shape[0],1))
        varinv = 1.0/variances
        res = jnp.dot(diff.T, varinv) / jnp.sum(varinv) * diff.shape[0]
        res = jnp.sum(res)/(jnp.sum(positions))
        return res
    
    @functools.partial(jit, static_argnums=(0))
    def loss_fn(self, logits:ArrayLike, targets:ArrayLike, masked=None, non_masked=None,variances: ArrayLike=None, alpha=1.0):
        masked_loss = self.masked_loss_fn(logits,targets,masked,variances) if masked is not None else 0
        loss = masked_loss
        if self.non_masked:
            unmasked_loss = self.masked_loss_fn(logits,targets,non_masked,variances) if non_masked is not None else 0
            loss = loss + alpha*unmasked_loss
        return loss
    
@dataclass(frozen=True)
class MWMLossMseNAndContrast(LossFcn):
    non_masked:bool=False
    temperature:float=1.0
    alpha:float=1.0
    beta:float=0.75
    
    @functools.partial(jit, static_argnums=(0))
    def masked_loss_fn(self,predictions, targets, positions,variances):
        diff = jnp.square(predictions-targets)
        diff = jnp.multiply(positions, diff) #shape [batch_size, num_chunks, chunk_size]
        diff = jnp.sum(diff, axis=(0,2)) #shape [num_chunks,]
        diff = jnp.reshape(diff, (diff.shape[0],1))
        varinv = 1.0/variances
        res = jnp.dot(diff.T, varinv) / jnp.sum(varinv) * diff.shape[0]
        res = jnp.sum(res)/(jnp.sum(positions))
        return res
    
    @functools.partial(jit, static_argnums=(0))
    def nt_xent_loss_fn(self, logits, masked_positions):
        z = jnp.multiply(logits, masked_positions)
        #z = logits
        z = z.reshape(z.shape[0], -1)
        z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)  # L2 normalize

        hidden1, hidden2 = jnp.split(z, 2, 0)
        batch_size = hidden1.shape[0]

        masks = jnp.eye(batch_size)
        labels = jnp.concatenate([jnp.eye(batch_size), jnp.zeros_like(masks)], axis=1) 
        
        logits_aa = jnp.matmul(hidden1, hidden1.T) / self.temperature - masks*1e9
        #logits_aa = jnp.where(masks, -jnp.inf, logits_aa)
        logits_bb = jnp.matmul(hidden2, hidden2.T) / self.temperature- masks*1e9
        #logits_bb = jnp.where(masks, -jnp.inf, logits_bb)
        logits_ab = jnp.matmul(hidden1, hidden2.T) / self.temperature
        logits_ba = jnp.matmul(hidden2, hidden1.T) / self.temperature

        log_prob_a = jax.nn.log_softmax(jnp.concatenate([logits_ab, logits_aa], axis=1))
        log_prob_b = jax.nn.log_softmax(jnp.concatenate([logits_ba, logits_bb], axis=1))
        
        loss_a = -jnp.mean(jnp.sum(labels * log_prob_a, axis=1))
        loss_b = -jnp.mean(jnp.sum(labels * log_prob_b, axis=1))

        #jax.debug.print("{x}",x=loss_a)
        #jax.debug.print("{x}",x=loss_b)
        loss = loss_a + loss_b
        return loss

    @functools.partial(jit, static_argnums=(0))
    def loss_fn(self, logits:ArrayLike, targets:ArrayLike, masked=None, non_masked=None,variances: ArrayLike=None, alpha=alpha,beta=beta):
        # reconstruction
        masked_loss = self.masked_loss_fn(logits,targets,masked,variances) if masked is not None else 0
        loss = masked_loss
        if self.non_masked:
            unmasked_loss = self.masked_loss_fn(logits,targets,non_masked,variances) if non_masked is not None else 0
            loss = loss + alpha*unmasked_loss
        
        #jax.debug.print("{}",loss)
        # contrastive  
        loss = beta*loss + (1-beta)*self.nt_xent_loss_fn(logits, masked)
        #jax.debug.print("{}",loss)
        return loss
    

@dataclass(frozen=True)
class MWMLossMseNAndContrastFromAux(LossFcn):
    non_masked:bool=False
    temperature:float=1.0
    alpha:float=1.0
    beta:float=0.75
    
    @functools.partial(jit, static_argnums=(0))
    def train_metrics(self, params, rng, data,state=None):
        inputs = data[0]
        targets = data[1]
        mask_positions = data[2]
        non_mask_positions = data[3]
        variances=data[4]
        if state is None:
            logits, aux = self.model_apply(params, rng, inputs, True) 
        else:
            (logits,aux), state=self.model_apply(params,state,rng,inputs,True)
            
        # reconstruction
        masked_loss = self.masked_loss_fn(logits,targets,mask_positions,variances) if mask_positions is not None else 0
        loss = masked_loss
        if self.non_masked:
            unmasked_loss = self.masked_loss_fn(logits,targets,non_mask_positions,variances) if non_mask_positions is not None else 0
            loss = loss + self.alpha*unmasked_loss
        # contrastive  
        loss = self.beta*loss + (1-self.beta)*self.nt_xent_loss_fn(aux, mask_positions)
        return loss, state
    
    @functools.partial(jit, static_argnums=(0))
    def val_metrics(self, params, rng, data,state=None):
        inputs = data[0]
        targets = data[1]
        mask_positions = data[2]
        non_mask_positions=data[3]
        variances=data[4]
        if state is None:
            logits,aux = self.model_apply(params, rng, inputs, False)
        else:
            (logits,aux),state = self.model_apply(params, state, rng, inputs, False)
        # reconstruction
        masked_loss = self.masked_loss_fn(logits,targets,mask_positions,variances) if mask_positions is not None else 0
        loss = masked_loss
        if self.non_masked: 
            unmasked_loss = self.masked_loss_fn(logits,targets,non_mask_positions,variances) if non_mask_positions is not None else 0
            loss = loss + self.alpha*unmasked_loss
        # contrastive  
        loss = self.beta*loss + (1-self.beta)*self.nt_xent_loss_fn(aux, mask_positions)
        return loss, state
    
    @functools.partial(jit, static_argnums=(0))
    def masked_loss_fn(self,predictions, targets, positions,variances):
        diff = jnp.square(predictions-targets)
        diff = jnp.multiply(positions, diff) #shape [batch_size, num_chunks, chunk_size]
        diff = jnp.sum(diff, axis=(0,2)) #shape [num_chunks,]
        diff = jnp.reshape(diff, (diff.shape[0],1))
        varinv = 1.0/variances
        res = jnp.dot(diff.T, varinv) / jnp.sum(varinv) * diff.shape[0]
        res = jnp.sum(res)/(jnp.sum(positions))
        return res
    
    @functools.partial(jit, static_argnums=(0))
    def nt_xent_loss_fn(self, logits, masked_positions):
        z = jnp.multiply(logits, masked_positions)
        #z = logits
        z = z.reshape(z.shape[0], -1)
        z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)  # L2 normalize

        hidden1, hidden2 = jnp.split(z, 2, 0)
        batch_size = hidden1.shape[0]

        masks = jnp.eye(batch_size)
        labels = jnp.concatenate([jnp.eye(batch_size), jnp.zeros_like(masks)], axis=1) 
        
        logits_aa = jnp.matmul(hidden1, hidden1.T) / self.temperature - masks*1e9
        #logits_aa = jnp.where(masks, -jnp.inf, logits_aa)
        logits_bb = jnp.matmul(hidden2, hidden2.T) / self.temperature- masks*1e9
        #logits_bb = jnp.where(masks, -jnp.inf, logits_bb)
        logits_ab = jnp.matmul(hidden1, hidden2.T) / self.temperature
        logits_ba = jnp.matmul(hidden2, hidden1.T) / self.temperature

        log_prob_a = jax.nn.log_softmax(jnp.concatenate([logits_ab, logits_aa], axis=1))
        log_prob_b = jax.nn.log_softmax(jnp.concatenate([logits_ba, logits_bb], axis=1))
        
        loss_a = -jnp.mean(jnp.sum(labels * log_prob_a, axis=1))
        loss_b = -jnp.mean(jnp.sum(labels * log_prob_b, axis=1))

        #jax.debug.print("{x}",x=loss_a)
        #jax.debug.print("{x}",x=loss_b)
        loss = loss_a + loss_b
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

    def cosine_distance(self,y_true, y_pred):
        y_true_norm = jnp.linalg.norm(y_true, axis=-1, keepdims=True)
        y_pred_norm = jnp.linalg.norm(y_pred, axis=-1, keepdims=True)
        dot_product = jnp.sum(y_true * y_pred, axis=-1, keepdims=True)
        cosine_distance = jnp.mean(1.0 - (dot_product / (y_true_norm * y_pred_norm)))
        return cosine_distance
    
    def loss_fn(self, logits, targets, masked=None, non_masked=None, variances=None,alpha=1.0):
        masked_loss = self.masked_loss_fn(logits,targets,masked) if masked is not None else 0
        loss = masked_loss
        if self.non_masked:
            unmasked_loss = self.masked_loss_fn(logits,targets,non_masked) if non_masked is not None else 0
            loss = loss + alpha*unmasked_loss
        return loss
    