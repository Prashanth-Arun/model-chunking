from setuptools import setup, find_packages

setup(
    name='model_chunking',
    version='0.0.1',
    description='Code for CS 854 F24 EOT Project - Model Chunking for Faster Inference <Working Title>"',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Dongfu Jiang',
    author_email='dongfu.jiang@uwaterloo.ca',
    packages=find_packages(),
    url='https://github.com/Prashanth-Arun/model-chunking.git',
    install_requires=[
        "transformers",
        "sentencepiece",
        "torch",
        "torch",
        "accelerate",
        "datasets==2.18.0"
    ],
    extras_require={
        "train": [
            "fire",
            "tqdm",
            "numpy",
            "requests",
            "matplotlib",
            "transformers_stream_generator",
            "chardet",
            "deepspeed",
            "peft>=0.10",
            "bitsandbytes",
            "wandb",
            "scipy",
            "webdataset",
            "pandas",
            "orjson",
            "prettytable",
            "pytest",
            "pyarrow",
            "dask",
            "einops-exts",
            "datasets==2.18.0",
        ],
        "eval": [
            "tqdm",
            "numpy",
            "prettytable",
            "fire",
            "datasets==2.18.0",
        ]
    }
)



# change it to pyproject.toml
# [build-system]
# python setup.py sdist bdist_wheel
# twine upload dist/*