from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaConfig, LlamaForCausalLM
import json
from tqdm import tqdm
import torch

model = AutoModelForCausalLM.from_pretrained('./llama-finetuned/checkpoint-936')
tokenizer = AutoTokenizer.from_pretrained('./llama-finetuned/checkpoint-936')

model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Add padding token if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return [{"input": item["prompt"], "output": item['label']} for item in tqdm(data)]


def test_model(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=10)
    return tokenizer.decode(output[0], skip_special_tokens=True)

file_path = "mnist_test_prompts.jsonl"
data = load_data(file_path)

print(data[11])
# exit()

test_prompt = data[11]['input']
print(test_model(test_prompt))

