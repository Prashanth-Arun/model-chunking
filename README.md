# model-chunking
Project for CS 854 (Model Serving Systems for GenAI), uWaterloo, Fall 2024.

## installation
```bash
# in your project env
conda create -n model-chunking python=3.10
conda activate model-chunking
pip install -e .
pip install flash-attn --no-build-isolation # [optional] for accelerated attention computation
```

## usage
```bash
python test_qwen2.py
```