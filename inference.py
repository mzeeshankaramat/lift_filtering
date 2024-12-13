from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import torch
import argparse


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
        generated_response = test_model(prompt)
        # generated_response = "Answer is 7"  # Example response
        results.append({"prompt": prompt, "label": label, "response": generated_response})
    
    # Save results to a JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

# Main function to handle argument parsing
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate responses for input prompts")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the generated responses")
    
    # Parse the arguments
    args = parser.parse_args()

    # Load test data from the input file
    data = load_data(args.input_file)

    # Generate responses and save to the output file
    generate_responses(data, args.output_file)

    print(f"Responses generated and saved to {args.output_file}")


if __name__ == "__main__":
    main()
