from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import torch
import argparse
import os

model = AutoModelForCausalLM.from_pretrained('./llama-finetuned/checkpoint-936')
tokenizer = AutoTokenizer.from_pretrained('./llama-finetuned/checkpoint-936')


model = model.to('cuda' if torch.cuda.is_available() else 'cpu')


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return [{"input": item["prompt"], "output": item['label']} for item in data]

def test_model(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=10)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_responses(data, output_file):
    results = []
    for item in tqdm(data, desc="Generating responses"):
        prompt = item["input"]
        label = item["output"]
        generated_response = test_model(prompt)
        results.append({"prompt": prompt, "label": label, "response": generated_response})
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Generate responses for input prompts")
    parser.add_argument('--epsilon', type=float, required=True, help="Epsilon value (e.g., 0.01, 0.05, etc.)")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the generated responses")
    
    args = parser.parse_args()

    epsilon_str = f"{args.epsilon:.2f}" 
    input_file = f'attacked_mnist_test_prompts_eps_{epsilon_str}.jsonl'

    if not os.path.exists(input_file):
        print(f"Error: The file {input_file} does not exist.")
        return

    data = load_data(input_file)

    generate_responses(data, args.output_file)

    print(f"Responses generated and saved to {args.output_file}")

if __name__ == "__main__":
    main()
