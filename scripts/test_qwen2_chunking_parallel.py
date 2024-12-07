from model_chunking.models.qwen2 import Qwen2ChunkingForCausalLM, Qwen2ChunkingConfig, Qwen2Tokenizer, Qwen2Config
from tqdm.auto import tqdm

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

config = Qwen2ChunkingConfig.from_pretrained(model_name, num_layers_per_chunk=8, chunking_mode="uniform_with_shared_start", aggregation_mode="mean", use_adapters=False)

# 2 gpus

device_map = {'model.embed_tokens': 0, 'lm_head': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 1, 'model.layers.10': 0, 'model.layers.11': 1, 'model.layers.12': 0, 'model.layers.13': 1, 'model.layers.14': 0, 'model.layers.15': 1, 'model.layers.16': 0, 'model.layers.17': 1, 'model.layers.18': 0, 'model.layers.19': 1, 'model.layers.20': 0, 'model.layers.21': 1, 'model.layers.22': 0, 'model.layers.23': 1, 'model.norm': 1, 'model.rotary_emb': 1, 'model.aggregation_head': 1} # uniform
model = Qwen2ChunkingForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype="auto",
    device_map=device_map
)

tokenizer = Qwen2Tokenizer.from_pretrained(model_name)

prompt = "Can you teach to how to use python to draw a turtle?"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=False
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
