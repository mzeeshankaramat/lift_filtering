import torch
import utils

def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return [{"input": item["prompt"], "output": item['label']} for item in data]

def magnitude_filtering(model, threshold):
    for name, module in model.named_modules():
        # Locate MLP layers (adjust this based on the LLaMA implementation)
        if hasattr(module, 'fc_in') and hasattr(module, 'fc_out'):
            print(f"Applying threshold to weights for MLP layer: {name}")
            with torch.no_grad():
                # Zero out weights below the threshold for fc_in
                module.fc_in.weight[module.fc_in.weight.abs() < threshold] = 0
                module.fc_in.bias[module.fc_in.bias.abs() < threshold] = 0
                # Zero out weights below the threshold for fc_out
                module.fc_out.weight[module.fc_out.weight.abs() < threshold] = 0
                module.fc_out.bias[module.fc_out.bias.abs() < threshold] = 0
    return model
                
def percentage_filtering(model, percentage):
    for i, layer in enumerate(model.model.decoder.layers):  # Adjust this based on LLaMA structure
        if hasattr(layer, 'mlp'):  # Ensure the layer has an MLP module
            print(f"Randomly zeroing out weights for MLP in layer {i}")
            with torch.no_grad():
                # Flatten weights to apply random masking
                fc_in_weights = layer.mlp.fc_in.weight.view(-1)
                fc_out_weights = layer.mlp.fc_out.weight.view(-1)
                num_fc_in_weights_to_zero = int(len(fc_in_weights) * percentage / 100)
                num_fc_out_weights_to_zero = int(len(fc_out_weights) * percentage / 100)

                # Generate random indices for zeroing
                fc_in_indices = random.sample(range(len(fc_in_weights)), num_fc_in_weights_to_zero)
                fc_out_indices = random.sample(range(len(fc_out_weights)), num_fc_out_weights_to_zero)

                # Zero out selected weights
                fc_in_weights[fc_in_indices] = 0
                fc_out_weights[fc_out_indices] = 0

                # Reshape weights back to their original shape
                layer.mlp.fc_in.weight.copy_(fc_in_weights.view(layer.mlp.fc_in.weight.size()))
                layer.mlp.fc_out.weight.copy_(fc_out_weights.view(layer.mlp.fc_out.weight.size()))
    return model

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
    pruning_percentages = [0.02, 0.04, 0.06, 0.08, 0.1]
    perturbations = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    num_examples = 100  # Number of examples to evaluate

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare to store accuracies for plotting
    filtered_accuracies = []

    # Loop over each pruning percentage
    for pruning_percentage in pruning_percentages:
        print(f"\n=== Pruning {pruning_percentage * 100:.2f}% of weights ===")

        # Prune the model
        pruned_model = percentage_filtering(filtered_model, pruning_percentage)
        pruned_model.to(device)  # Ensure the pruned model is also on the right device

        # Store accuracies for the current pruned model
        accuracy_filtered = []

        for perturbation in perturbations:
            print(f"\n--- Perturbation {perturbation * 100:.2f}% ---")

            epsilon_str = perturbation
            input_file = f'attacked_mnist_test_prompts_eps_{epsilon_str}.jsonl'

            if not os.path.exists(input_file):
                print(f"Error: The file {input_file} does not exist.")
                return

            data = load_data(input_file)

            generate_responses(data, "output.json")
            
            accuracy_filtered_after_attack = calculate_accuracy("output.json")
            
            # Store the filtered accuracy
            accuracy_filtered.append(accuracy_filtered_after_attack)

            print(f"Filtered Model Accuracy (After Attack): {accuracy_filtered_after_attack * 100:.2f}%")

        filtered_accuracies.append(accuracy_filtered)

        # Clear GPU cache between iterations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()