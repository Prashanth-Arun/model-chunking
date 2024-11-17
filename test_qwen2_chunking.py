from model_chunking.models.qwen2 import Qwen2ChunkingForCausalLM, Qwen2ChunkingConfig, Qwen2Tokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

config = Qwen2ChunkingConfig.from_pretrained(model_name, num_layers_per_chunk=3, chunking_mode="sequential", aggregation_mode="mean")


model = Qwen2ChunkingForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = Qwen2Tokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
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
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)