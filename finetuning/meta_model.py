import haiku as hk
import jax
import jax.tree_util as tree_util
import jax.numpy as jnp
from jax.typing import ArrayLike

from model.meta_model import create_meta_model_classifier, MetaModel, Transformer, MetaModelClassifier
from model.meta_model import MetaModelConfig, MetaModelClassifierConfig
from model_zoo_jax import model_restore, TrainState

def get_meta_model_fcn(config:MetaModelConfig, num_classes:int, model_type:str,compress_size:int=1200) -> hk.Transformed:
    if model_type=="classifier":
        # meta model classifier, with ViT inspired architecture, takes only output[:,0,:]
        classifier_config = MetaModelClassifierConfig(
            model_size=config.model_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout_rate=config.dropout_rate,
            use_embedding=config.use_embedding,
            num_classes=num_classes
            )
        model = create_meta_model_classifier(classifier_config)
    elif model_type=="classifier_compress":
        @hk.transform
        def model(input_batch: dict,
                is_training: bool = True) -> ArrayLike:
            net = MetaModelClassifier(
                model_size=config.model_size,
                num_classes=compress_size,
                use_embedding=config.use_embedding,
                transformer=Transformer(
                    num_heads=config.num_heads,
                    num_layers=config.num_layers,
                    key_size=config.key_size,
                    dropout_rate=config.dropout_rate,
                    widening_factor=1000/512,
                    activation='leakyrelu'
                ))
            linear = hk.Linear(num_classes)
            x = net(input_batch, is_training=is_training)
            x = hk.Flatten()(x)
            y = linear(x)
            return y
    elif model_type=="plus_linear":
        # take the whole meta model architecture and add another layer on top
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
            linear = hk.Linear(num_classes)
            x = net(input_batch, is_training=is_training)
            x = hk.Flatten()(x)
            y = linear(x)
            return y
        
    elif model_type=="replace_last":
        # replace the last layer with a new linear layer
        if config.use_embedding:
        
            # TODO: hard coded for now - recognize replaced embedding layer
            def f(x):
                linear =hk.Sequential([lambda a: hk.Linear(config.model_size)(a)],name="helper_name")
                return linear(x)
                
            @hk.transform
            def model(input_batch: dict,
                    is_training: bool = True) -> ArrayLike:
                net = MetaModel(
                    model_size=config.model_size,
                    use_embedding=False,
                    max_seq_len=config.max_seq_len,
                    transformer=Transformer(
                        num_heads=config.num_heads,
                        num_layers=config.num_layers,
                        key_size=config.key_size,
                        dropout_rate=config.dropout_rate,
                    )) 
                
                input_batch = f(input_batch)
                x = net(input_batch, is_training=is_training)
                x = hk.Flatten()(x)
                y = hk.Linear(num_classes)(x)
                return y
    elif model_type=="avgpool":
        # replace the last layer with a new linear layer
        if config.use_embedding:
        
            # TODO: hard coded for now - recognize replaced embedding layer
            def f(x):
                linear =hk.Sequential([lambda a: hk.Linear(config.model_size)(a)],name="helper_name")
                return linear(x)
                
            @hk.transform
            def model(input_batch: dict,
                    is_training: bool = True) -> ArrayLike:
                net = MetaModel(
                    model_size=config.model_size,
                    use_embedding=False,
                    max_seq_len=config.max_seq_len,
                    transformer=Transformer(
                        num_heads=config.num_heads,
                        num_layers=config.num_layers,
                        key_size=config.key_size,
                        dropout_rate=config.dropout_rate,
                    )) 
                
                input_batch = f(input_batch)
                x = net(input_batch, is_training=is_training)
                x = jnp.mean(x,axis=1)
                y = hk.Linear(num_classes)(x)
                return y
        else:
            @hk.transform
            def model(input_batch: dict,
                    is_training: bool = True) -> ArrayLike:
                net = MetaModel(
                    model_size=config.model_size,
                    use_embedding=False,
                    max_seq_len=config.max_seq_len,
                    transformer=Transformer(
                        num_heads=config.num_heads,
                        num_layers=config.num_layers-1, #replace the last transformer layer
                        key_size=config.key_size,
                        dropout_rate=config.dropout_rate,
                    )) 
                x = net(input_batch, is_training=is_training)
                x = hk.Flatten()(x)
                y = hk.Linear(num_classes)(x)
                return y
    else:
        ValueError("Model type not valid")
    
    return model

def load_pretrained_meta_model_parameters(params,path:str):
    print('Loading parameters from: ',path)
    pretrained_params = model_restore(path)
    flat_pretrained_params = tree_util.tree_flatten_with_path(pretrained_params)[0]
    #print(tree_util.tree_map(lambda x:x.shape, pretrained_params))
    #print(tree_util.tree_map(lambda x:x.shape, params))
    
    def get_param_from_keypath(key, tree=flat_pretrained_params):
        # TODO: hard coded for now - recognize replaced embedding layer
        key = tree_util.tree_map(lambda k: tree_util.DictKey(key=k.key.replace('helper_name', 'meta_model')) if k.key.startswith('helper_name') else k, key)
        for keypath, param in tree:
            keypath = tree_util.tree_map(lambda k: tree_util.DictKey(key=k.key.replace('meta_model_with_aux', 'meta_model')), keypath)
            keypath = tree_util.tree_map(lambda k: tree_util.DictKey(key=k.key.replace('transformer_with_aux', 'transformer')), keypath)
            if key==keypath:
                return param
        return None
    
    def param_update(key, param):
        newparam = get_param_from_keypath(key)
        if newparam is not None and jnp.shape(param)==jnp.shape(newparam):
            #print(key, 'loaded')
            return newparam  # Replace param1 with param2 value
        else:
            if newparam is not None:
                print(jnp.shape(newparam),jnp.shape(param))
            print(key, 'initialized')
            return param  # Keep param1 value as it is
    
    loaded_params = tree_util.tree_map_with_path(param_update, params)
    return loaded_params

def load_pretrained_state(state, path:str):
    params = load_pretrained_meta_model_parameters(state.params,path)
    return TrainState(step=state.step, 
                      rng=state.rng, 
                      opt_state=state.opt_state, 
                      params=params, 
                      model_state=state.model_state)
    