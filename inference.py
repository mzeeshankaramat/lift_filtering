from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained('./llama-finetuned/checkpoint-936')
tokenizer = AutoTokenizer.from_pretrained('./llama-finetuned/checkpoint-936')

# Move model to appropriate device
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Add padding token if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Function to load data
def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return [{"input": item["prompt"], "output": item['label']} for item in data]

# Function to test the model on a single prompt
def test_model(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=10)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Function to generate responses for all inputs
def generate_responses(data, output_file):
    results = []
    for item in tqdm(data, desc="Generating responses"):
        prompt = item["input"]
        label = item["output"]
        # generated_response = test_model(prompt)
        generated_response = "Answer is 7"
        results.append({"prompt": prompt, "label": label, "response": generated_response})
    
    # Save results to a JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

# Load test data
file_path = "mnist_test_prompts.jsonl"
data = load_data(file_path)

# Generate responses and save to file
output_file = "generated_responses.json"
generate_responses(data, output_file)

print(f"Responses generated and saved to {output_file}")


