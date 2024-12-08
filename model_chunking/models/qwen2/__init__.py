from .modeling_qwen2 import Qwen2ForCausalLM, Qwen2ChunkingForCausalLM, chunking_layers
from .tokenization_qwen2 import Qwen2Tokenizer
from .configuration_qwen2 import Qwen2Config, Qwen2ChunkingConfig

from transformers import AutoConfig, AutoModelForCausalLM


AutoConfig.register("qwen2_chunking", Qwen2ChunkingConfig)
AutoModelForCausalLM.register(Qwen2ChunkingConfig, Qwen2ChunkingForCausalLM)

import torch

def infer_chunking_device_map(num_total_layers, num_layers_per_chunk, num_chunks, chunking_mode, num_gpus):
    num_available_devices = torch.cuda.device_count()
    if num_gpus > num_available_devices:
        raise ValueError(f"num_gpus ({num_gpus}) is greater than the number of available devices ({num_available_devices})")
    layers = list(range(num_total_layers))
    all_chunking_layers, pre_chunking_layers, post_chunking_layers = chunking_layers(
        layers, chunking_mode, num_layers_per_chunk, num_chunks
    )
    device_map = {}
    availables_devices = list(range(num_gpus))
    # assign try to assign chunking layers to different gpus
    num_non_chunking_layers = len(pre_chunking_layers) + len(post_chunking_layers)
    num_non_chunking_layers_per_gpu = (num_non_chunking_layers - 1) // num_gpus + 1
    for i, layer in enumerate(pre_chunking_layers+post_chunking_layers):
        device_map[f"model.layers.{layer}"] = availables_devices[i // (num_non_chunking_layers_per_gpu)]
    for i, chunk_layers in enumerate(all_chunking_layers):
        for layer in chunk_layers:
            device_map[f"model.layers.{layer}"] = availables_devices[i % len(availables_devices)]
    device_map["model.embed_tokens"] = availables_devices[0]
    device_map["lm_head"] = availables_devices[0]
    device_map["model.norm"] = availables_devices[-1]
    device_map["model.rotary_emb"] = availables_devices[-1]
    device_map["model.aggregation_head"] = availables_devices[-1]
    return device_map

def infer_chunking_device_map_from_config(config, num_gpus):
    return infer_chunking_device_map(
        num_total_layers=config.num_hidden_layers,
        num_layers_per_chunk=config.num_layers_per_chunk,
        num_chunks=config.num_chunks,
        chunking_mode=config.chunking_mode,
        num_gpus=num_gpus
    )