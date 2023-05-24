from setuptools import setup, find_packages

setup(
    name = 'mmfi',
    version = '0.0.1',
    url = '',
    description = 'Meta models for interpretability',
    packages=["pretraining"],
    install_requires=[
        "chex",
        "datasets",
        "dm-haiku",
        "jax",
        "matplotlib",
        "numpy",
        "optax",
        "pandas",
        "wandb",
        "pytest",
        # Other github repos
        'meta-transformer @ git+ssh://git@github.com:langosco/meta-transformer.git',
        'neural-net-augmentations @ git+ssh://git@github.com:Emilija2000/neural-net-augmentation.git',
        'model-zoo @ git+ssh://git@github.com:Emilija2000/model-zoo-jax.git'
    ]
)
