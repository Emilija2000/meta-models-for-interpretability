from dataclasses import dataclass
import chex
import functools
import jax
from jax import jit, value_and_grad
import jax.numpy as jnp
from jax.random import PRNGKey
import optax
from optax import OptState
from typing import Callable, Tuple,Optional

from pretraining.loss import LossFcn

@chex.dataclass
class TrainState:
    step: int
    rng: PRNGKey
    opt_state: OptState
    params: dict
    model_state: Optional[Tuple[jnp.array]] = None
    
@dataclass(frozen=True)  # needs to be immutable to be hashable
class Updater: 
    """Holds training methods. All methods are jittable."""
    opt: optax.GradientTransformation
    evaluator: LossFcn
    model_init: Callable

    @functools.partial(jit, static_argnums=(0))
    def init_params(self, rng: jnp.ndarray, x:jnp.ndarray) -> dict:
        """Initializes state of the updater."""
        out_rng, k0 = jax.random.split(rng)
        params = self.model_init(k0, x, is_training=True)
        if isinstance(params, tuple) and len(params)==2:
            params,model_state=params
        else:
            model_state=None
        opt_state = self.opt.init(params)
        
        return TrainState(
            step=0,
            rng=out_rng,
            opt_state=opt_state,
            params=params,
            model_state=model_state
        )
    
    @functools.partial(jit, static_argnums=0)
    def train_step(self, state: TrainState, data:dict) -> Tuple[TrainState, dict]:
        state.rng, *subkeys = jax.random.split(state.rng, 3)
        
        (loss, new_state), grads = value_and_grad(self.evaluator.train_metrics, has_aux=True)(state.params, subkeys[1], data,state.model_state)
    
        state.model_state=new_state
        updates, state.opt_state = self.opt.update(grads, state.opt_state, state.params)
        state.params = optax.apply_updates(state.params, updates)
        state.step += 1
        metrics = {
                "train/loss": loss,
                "step": state.step,
        }
        return state, metrics

    @functools.partial(jit, static_argnums=0)
    def val_step(self, state: TrainState, data: dict) -> Tuple[TrainState, dict]:
        state.rng, subkey = jax.random.split(state.rng)
        loss, state.model_state = self.evaluator.val_metrics(state.params, subkey, data,state.model_state)
        metrics = {
                "val/loss": loss
        }
        return state, metrics