import dataclasses
from typing import Optional
import haiku as hk
import jax
import jax.numpy as jnp
from model.transformer import Transformer, TransformerConfig, TransformerWithAux
from jax.typing import ArrayLike
import chex

@dataclasses.dataclass
class MetaModelClassifier(hk.Module):
    """A simple meta-model."""

    transformer: Transformer
    model_size: int
    num_classes: int
    name: Optional[str] = None
    use_embedding: Optional[bool] = False

    def __call__(
        self,
        inputs: ArrayLike,  # dict
        *,
        is_training: bool = True,
    ) -> jax.Array:
        """Forward pass. Returns a sequence of logits."""
        if self.use_embedding:
            inputs = hk.Linear(self.model_size)(inputs)
        batch_size, seq_len, _ = inputs.shape

        # Add classification token.
        cls_token = hk.get_parameter(
            'cls_token', [1, 1, self.model_size], init=jnp.zeros)
        cls_tokens = jnp.tile(cls_token, [batch_size, 1, 1])  # [B, 1, D]
        inputs = jnp.concatenate([cls_tokens, inputs], axis=1)  # [B, T+1, D]

        # Add positional embeddings.
        init = hk.initializers.TruncatedNormal(stddev=0.02)
        positional_embeddings = hk.get_parameter(
            'positional_embeddings', [seq_len + 1, self.model_size], init=init)
        inputs = inputs + positional_embeddings  # [B, T+1, D]

        # Run the transformer over the inputs.
        outputs = self.transformer(inputs, is_training=is_training)

        cls_out = outputs[:, 0, :]  # [B, D]
        return hk.Linear(self.num_classes, name="linear_output")(cls_out)  # [B, V]


@chex.dataclass
class MetaModelClassifierConfig(TransformerConfig):
    """Hyperparameters for the model."""
    num_classes: int = 4
    use_embedding: bool = False


def create_meta_model_classifier(
        config: MetaModelClassifierConfig) -> hk.Transformed:
    @hk.transform
    def model(params_batch: dict, 
              is_training: bool = True) -> ArrayLike:
        net = MetaModelClassifier(
            model_size=config.model_size,
            num_classes=config.num_classes,
            use_embedding=config.use_embedding,
            transformer=Transformer(
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                key_size=config.key_size,
                dropout_rate=config.dropout_rate,
            ))
        return net(params_batch, is_training=is_training)
    return model


@dataclasses.dataclass
class MetaModel(hk.Module):
    """A meta-model that returns neural network parameters."""

    transformer: Transformer
    model_size: int
    name: Optional[str] = None
    use_embedding: Optional[bool] = False
    max_seq_len: Optional[int] = 500

    def __call__(
            self,
            inputs: ArrayLike,
            *,
            is_training: bool = True
        ) -> jax.Array:
        """Forward pass. Returns a sequence of output embeddings."""
        input_shape = inputs.shape[-1]
        if self.use_embedding:
            inputs = hk.Linear(self.model_size)(inputs)
        _, seq_len, _ = inputs.shape

        # Add positional embeddings.
        if self.max_seq_len == None:
            init = hk.initializers.TruncatedNormal(stddev=0.02)
            positional_embeddings = hk.get_parameter('positional_embeddings', [seq_len, self.model_size], init=init)
        else:
            init = hk.initializers.TruncatedNormal(stddev=0.02)
            positional_embeddings = hk.get_parameter('positional_embeddings', [self.max_seq_len, self.model_size], init=init)
        
        inputs = inputs + jnp.tile(positional_embeddings[:seq_len, :],(inputs.shape[0],1,1))

        # Run the transformer over the inputs.
        outputs = self.transformer(
            inputs, is_training=is_training)  # [B, T, D]

        if self.use_embedding:
            outputs = hk.Linear(input_shape)(outputs)

        return outputs
    
@dataclasses.dataclass
class MetaModelWithAux(hk.Module):
    """A meta-model that returns neural network parameters."""

    transformer: TransformerWithAux
    model_size: int
    name: Optional[str] = None
    use_embedding: Optional[bool] = False
    max_seq_len: Optional[int] = 500

    def __call__(
            self,
            inputs: ArrayLike,
            *,
            is_training: bool = True
        ) -> jax.Array:
        """Forward pass. Returns a sequence of output embeddings."""
        input_shape = inputs.shape[-1]
        if self.use_embedding:
            inputs = hk.Linear(self.model_size)(inputs)
        _, seq_len, _ = inputs.shape

        # Add positional embeddings.
        if self.max_seq_len == None:
            init = hk.initializers.TruncatedNormal(stddev=0.02)
            positional_embeddings = hk.get_parameter('positional_embeddings', [seq_len, self.model_size], init=init)
        else:
            init = hk.initializers.TruncatedNormal(stddev=0.02)
            positional_embeddings = hk.get_parameter('positional_embeddings', [self.max_seq_len, self.model_size], init=init)
        
        inputs = inputs + jnp.tile(positional_embeddings[:seq_len, :],(inputs.shape[0],1,1))

        # Run the transformer over the inputs.
        outputs,aux = self.transformer(
            inputs, is_training=is_training)  # [B, T, D]

        if self.use_embedding:
            outputs = hk.Linear(input_shape)(outputs)
            aux = hk.Linear(input_shape)(aux)
        return outputs,aux


@chex.dataclass
class MetaModelConfig(TransformerConfig):
    """Hyperparameters for the model."""
    use_embedding: Optional[bool] = False
    max_seq_len: Optional[int] = None


def create_meta_model(
        config: MetaModelConfig) -> hk.Transformed:
    @hk.transform
    def model(input_batch: dict,
              is_training: bool = True) -> ArrayLike:
        net = MetaModel(
            model_size=config.model_size,
            use_embedding=config.use_embedding,
            max_seq_len=config.max_seq_len,
            transformer=Transformer(
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                key_size=config.key_size,
                dropout_rate=config.dropout_rate,
            ))
        return net(input_batch, is_training=is_training)
    return model

def create_meta_model_withaux(
        config: MetaModelConfig) -> hk.Transformed:
    @hk.transform
    def model(input_batch: dict,
              is_training: bool = True) -> ArrayLike:
        net = MetaModelWithAux(
            model_size=config.model_size,
            use_embedding=config.use_embedding,
            max_seq_len=config.max_seq_len,
            transformer=TransformerWithAux(
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                key_size=config.key_size,
                dropout_rate=config.dropout_rate,
            ))
        return net(input_batch, is_training=is_training)
    return model
