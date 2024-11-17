# model-chunking
Project for CS 854 (Model Serving Systems for GenAI), uWaterloo, Fall 2024.

## installation
```bash
# in your project env
conda create -n model-chunking python=3.10
conda activate model-chunking
pip install -e .
pip install flash-attn --no-build-isolation
```

## usage
```bash
python test_qwen2.py
```

## Training
### setup
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
```
Then add `import model_chunking` to `LLaMA-Factory/src/llamafactory/launcher.py`, it looks like this:
```python
import model_chunking # our models registered here
from llamafactory.train.tuner import run_exp  # use absolute import


def launch():
    run_exp()


if __name__ == "__main__":
    launch()
```

Then we can use our custom chunking models in LLaMA-Factory.

- Lora
```bash
llamafactory-cli train ./LLaMA-Factory/examples/train_lora/llama3_lora_sft.yaml
```

