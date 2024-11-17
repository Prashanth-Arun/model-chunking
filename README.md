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
cd LLaMA-Factory && pip install -e .
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

In the `./LLaMA-Factory/examples/train_lora/llama3_lora_sft.yaml`, change the `model_name_or_path` to be `DongfuJiang/Qwen2.5-0.5B-Instruct`. Then we can train the model.

- Lora
```bash
cd LLaMA-Factory && llamafactory-cli train ../training_configs/qwen2_chunking_lora.yaml
```

### How is `DongfuJiang/Qwen2.5-0.5B-Instruct` configured?
I simply change the `config.json` after duplicate the `Qwen/Qwen2.5-0.5B-Instruct`

```json
{
  "architectures": [
    "Qwen2ChunkingForCausalLM" # before it was "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "model_type": "qwen2_chunking", # before it was "qwen2"
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

To add other custom configs, add them like this:
```json
{
    ...,
    "num_layers_per_chunk": 3,
    "chunking_mode": "sequential",
    "aggregation_mode": "mean",
    "use_adapters": false
}
```
Then apply the changes to your huggingface model.