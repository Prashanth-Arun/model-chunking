from .modeling_qwen2 import Qwen2ForCausalLM, Qwen2ChunkingForCausalLM 
from .tokenization_qwen2 import Qwen2Tokenizer
from .configuration_qwen2 import Qwen2Config, Qwen2ChunkingConfig

from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("qwen2_chunking", Qwen2ChunkingConfig)
AutoModelForCausalLM.register(Qwen2ChunkingConfig, Qwen2ChunkingForCausalLM)