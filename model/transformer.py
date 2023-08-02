"""Base Transformer.

Glossary of shapes:
- B: Batch size.
- T: Sequence length.
- D: Model embedding size = d_model.
- H: Number of attention heads.
"""

import dataclasses
from typing import Optional

import haiku as hk
import jax
import chex


def layer_norm(x: jax.Array) -> jax.Array:
  """Applies a unique LayerNorm to x with default settings."""
  ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
  return ln(x)


@chex.dataclass
class TransformerConfig:
    """Hyperparameters for the model."""
    num_heads: int = 4
    num_layers: int = 2
    dropout_rate: float = 0.1
    model_size: int = 128

    def __post_init__(self):
        self.key_size = self.model_size // self.num_heads
        if self.model_size % self.num_heads != 0:
            raise ValueError(
                f"model_size ({self.model_size}) must be "
                "divisible by num_heads ({self.num_heads})")



@dataclasses.dataclass
class Transformer(hk.Module):
  """A transformer stack."""

  num_heads: int
  num_layers: int
  key_size: int
  dropout_rate: float
  widening_factor: Optional[int] = 4
  name: Optional[str] = None
  activation: Optional[str] = 'gelu'

  def __call__(
      self,
      embeddings: jax.Array,  # [B, T, D]
      *,
      is_training: bool = True,
  ) -> jax.Array:  # [B, T, D]
    """Transforms input embedding sequences to output embedding sequences."""

    initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
    dropout_rate = self.dropout_rate if is_training else 0.
    _, _, model_size = embeddings.shape

    h = embeddings
    for _ in range(self.num_layers):
      # First the attention block.
      attn_block = hk.MultiHeadAttention(
          num_heads=self.num_heads,
          key_size=self.key_size,
          model_size=model_size,
          w_init=initializer,
      )
      h_norm = layer_norm(h)
      h_attn = attn_block(h_norm, h_norm, h_norm)
      h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
      h = h + h_attn

      if self.activation == 'gelu':
          act = jax.nn.gelu
      else:
          act = jax.nn.leaky_relu
      # Then the dense block.
      dense_block = hk.Sequential([
          hk.Linear(int(self.widening_factor * model_size), w_init=initializer),
          act,
          hk.Linear(model_size, w_init=initializer),
      ])
      h_norm = layer_norm(h)
      h_dense = dense_block(h_norm)
      h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
      h = h + h_dense

    return layer_norm(h)
  
  
@dataclasses.dataclass
class TransformerWithAux(Transformer):
  """A transformer stack."""

  def __call__(
      self,
      embeddings: jax.Array,  # [B, T, D]
      *,
      is_training: bool = True,
  ) -> jax.Array:  # [B, T, D]
    """Transforms input embedding sequences to output embedding sequences."""

    initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
    dropout_rate = self.dropout_rate if is_training else 0.
    _, _, model_size = embeddings.shape

    h = embeddings
    for l in range(self.num_layers):
      # First the attention block.
      attn_block = hk.MultiHeadAttention(
          num_heads=self.num_heads,
          key_size=self.key_size,
          model_size=model_size,
          w_init=initializer,
      )
      h_norm = layer_norm(h)
      h_attn = attn_block(h_norm, h_norm, h_norm)
      h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
      h = h + h_attn

      # Then the dense block.
      dense_block = hk.Sequential([
          hk.Linear(int(self.widening_factor * model_size), w_init=initializer),
          #jax.nn.gelu,
          jax.nn.leakyrelu,
          hk.Linear(model_size, w_init=initializer),
      ])
      h_norm = layer_norm(h)
      h_dense = dense_block(h_norm)
      h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
      h = h + h_dense
      
      if l == self.num_layers-2:
        aux = h.copy()

    return layer_norm(h),layer_norm(aux)