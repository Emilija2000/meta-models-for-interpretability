from model_zoo_jax import load_nets
from pretraining import process_batch

from jax.random import PRNGKey, split
import jax.numpy as jnp

def test_mask_chunk(inputs):
    MASK_TOKEN = 0
    rng = PRNGKey(42)
    perc = 0.15
    chunk_s = 100
    
    # check portion of masked chunks - no indicators
    for per in [0.1,0.15,0.25,0.5,0.6]:
        rng, subkey = split(rng)
        masked_ins, labels, positions = process_batch(subkey, inputs, MASK_TOKEN, per, chunk_s, mask_individual=False, mask_indicators=False)
        for pos in positions:
            portion = jnp.sum(pos)/jnp.prod(jnp.array(jnp.shape(pos)))
            assert portion < perc+0.5
            assert portion > perc-0.5

    # check dimensions - indicators
    masked_ins, labels, positions = process_batch(subkey, inputs, MASK_TOKEN, perc, chunk_s, mask_individual=False, mask_indicators=True)
    assert masked_ins.shape[0] == len(inputs)
    assert labels.shape[0] == len(inputs)
    assert positions.shape[0] == len(inputs)
    assert masked_ins.shape[2] == chunk_s + 1
    assert labels.shape[2] == chunk_s + 1
    assert positions.shape[2] == chunk_s + 1
    # input and label padding
    assert jnp.sum(positions[:,:,-1]) == 0
    assert jnp.sum(positions)/(positions.shape[2]-1) == jnp.sum(masked_ins[:,:,-1])
    assert jnp.sum(labels[:,:,-1]) ==  0.0 #labels.shape[0]*labels.shape[1]
        
    # TODO:check mask mask token?
    
    # check portion of masked chunks - no indicators
    for per in [0.1,0.15,0.25,0.5,0.6]:
        rng, subkey = split(rng)
        masked_ins, labels, positions = process_batch(subkey, inputs, MASK_TOKEN, per, chunk_s, mask_individual=False, mask_indicators=False)
        for pos in positions:
            shape = jnp.array(jnp.shape(pos))
            size_without_padded = jnp.prod(shape) - shape[0]
            portion = jnp.sum(pos)/size_without_padded
            assert portion < perc+0.5
            assert portion > perc-0.5
    # check dimensions - no indicators
    masked_ins, labels, positions = process_batch(subkey, inputs, MASK_TOKEN, perc, chunk_s, mask_individual=False, mask_indicators=False)
    assert masked_ins.shape[0] == len(inputs)
    assert labels.shape[0] == len(inputs)
    assert positions.shape[0] == len(inputs)
    assert masked_ins.shape[2] == chunk_s 
    assert labels.shape[2] == chunk_s 
    assert positions.shape[2] == chunk_s 

    #check mask 0 percent
    masked_ins, labels, positions = process_batch(subkey, inputs, MASK_TOKEN,0, chunk_s, mask_individual=False, mask_indicators=True)
    assert jnp.sum(positions)==0.0
    print
    assert jnp.sum(masked_ins[:,:,-1])==0.0

def test_mask_individual(inputs):
    MASK_TOKEN = 0
    rng = PRNGKey(42)
    perc = 0.15
    chunk_s = 100
    
    # check portion of masked chunks - no indicators
    for per in [0.1,0.15,0.25,0.5,0.6]:
        rng, subkey = split(rng)
        masked_ins, labels, positions = process_batch(subkey, inputs, MASK_TOKEN, per, chunk_s, mask_individual=True, mask_indicators=False)
        for pos in positions:
            portion = jnp.sum(pos)/jnp.prod(jnp.array(jnp.shape(pos)))
            assert portion < perc+0.5
            assert portion > perc-0.5

    # check dimensions - indicators
    masked_ins, labels, positions = process_batch(subkey, inputs, MASK_TOKEN, perc, chunk_s, mask_individual=True, mask_indicators=True)
    assert masked_ins.shape[0] == len(inputs)
    assert labels.shape[0] == len(inputs)
    assert positions.shape[0] == len(inputs)
    assert masked_ins.shape[2] == chunk_s *2
    assert labels.shape[2] == chunk_s *2
    assert positions.shape[2] == chunk_s *2
    # input and label padding
    assert jnp.sum(positions[:,:,-1]) == 0
    assert jnp.sum(positions) == jnp.sum(masked_ins[:,:,chunk_s:])
    assert jnp.sum(labels[:,:,-1]) == 0.0 #labels.shape[0]*labels.shape[1]
        
    # check mask mask token?
    for m in [0, 1, 0.5]:
        masked_ins, labels, positions = process_batch(subkey, inputs, m, perc, chunk_s, mask_individual=True, mask_indicators=True)
        assert jnp.sum(masked_ins[positions]) == jnp.sum(positions)*m
    
    # check portion of masked chunks - no indicators
    for per in [0.1,0.15,0.25,0.5,0.6]:
        rng, subkey = split(rng)
        masked_ins, labels, positions = process_batch(subkey, inputs, MASK_TOKEN, per, chunk_s, mask_individual=True, mask_indicators=False)
        for pos in positions:
            shape = jnp.array(jnp.shape(pos))
            size_without_padded = jnp.prod(shape) - shape[0]
            portion = jnp.sum(pos)/size_without_padded
            assert portion < perc+0.5
            assert portion > perc-0.5
    # check dimensions - no indicators
    masked_ins, labels, positions = process_batch(subkey, inputs, MASK_TOKEN, perc, chunk_s, mask_individual=True, mask_indicators=False)
    assert masked_ins.shape[0] == len(inputs)
    assert labels.shape[0] == len(inputs)
    assert positions.shape[0] == len(inputs)
    assert masked_ins.shape[2] == chunk_s 
    assert labels.shape[2] == chunk_s 
    assert positions.shape[2] == chunk_s 
    
    #check mask 0 percent
    masked_ins, labels, positions = process_batch(subkey, inputs, MASK_TOKEN,0, chunk_s, mask_individual=True, mask_indicators=True)
    assert jnp.sum(positions)==0.0
    assert jnp.sum(masked_ins[:,:,chunk_s:])==0.0

if __name__=='__main__':
    inputs, _ = load_nets(n=3, 
                        data_dir='/rds/user/ed614/hpc-work/model_zoo_datasets/mnist_smallCNN_fixed_zoo',
                        flatten=False,
                        num_checkpoints=1)
    
    test_mask_chunk(inputs)
    test_mask_individual(inputs)

        