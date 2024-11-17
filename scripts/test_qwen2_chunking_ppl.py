from model_chunking.models.qwen2 import Qwen2ChunkingForCausalLM, Qwen2ChunkingConfig, Qwen2Tokenizer, Qwen2Config, Qwen2ForCausalLM
from tqdm.auto import tqdm
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
original_model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
original_config = Qwen2Config.from_pretrained(model_name)

# Divisors for Qwen2.5-0.5B = 2, 3, 4, 6, 8, 12
# divisor = 1 should decompose to the original model
f = open("logs/test_qwen2_chunking_ppl.txt", "w+")
divisors = [1, 2, 3, 4, 6, 8, 12, 24]
for divisor in tqdm(divisors):
    config = Qwen2ChunkingConfig.from_pretrained(model_name, num_layers_per_chunk=original_config.num_hidden_layers // divisor, chunking_mode="uniform", aggregation_mode="mean")

    model = Qwen2ChunkingForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name)

    prompt = "Tell me a short story."
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

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
        )

        full_sequence = torch.cat((model_inputs['input_ids'], generated_ids), dim=1)

        outputs = original_model(input_ids=full_sequence, labels=full_sequence)
        loss = outputs.loss
        perplexity = torch.exp(loss)

    response = tokenizer.batch_decode(generated_ids[:, model_inputs['input_ids'].shape[-1]:], skip_special_tokens=True)[0]
    print("-" * 20)
    print(f"Divisor = {divisor}")
    print(f"Perplexity: {perplexity.item()}")
    print("-" * 20)
    print(response)
    print("-" * 20)

    f.write("-" * 20 + "\n")
    f.write(f"Divisor = {divisor}\n")
    f.write(f"Perplexity: {perplexity.item()}\n")
    f.write("-" * 20 + "\n")
    f.write(response + "\n")
    f.write("-" * 20 + "\n")
    f.write("\n")