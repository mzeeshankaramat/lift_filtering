import torch
from torchvision import datasets, transforms
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import pandas as pd
import ujson


# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Use only 1000 samples for training and testing
# train_subset_size = 1000
# test_subset_size = 200

# mnist_train.data = mnist_train.data[:train_subset_size]
# mnist_train.targets = mnist_train.targets[:train_subset_size]
# mnist_test.data = mnist_test.data[:test_subset_size]
# mnist_test.targets = mnist_test.targets[:test_subset_size]

# Function to convert image to text prompt
def image_to_prompt(image, label):
    # Flatten the image and create a string representation
    flattened_image = image.numpy().flatten()
    pixel_values = " ".join(map(str, flattened_image))
    # Create a prompt with an initial and end phrase
    prompt = f"Given image with pixels {pixel_values}. What is this digit?"
    return {"prompt": prompt, "label": label}

# Convert dataset to prompts
train_prompts = [image_to_prompt(image, label) for image, label in zip(mnist_train.data, mnist_train.targets)]
test_prompts = [image_to_prompt(image, label) for image, label in zip(mnist_test.data, mnist_test.targets)]

# Convert prompts to DataFrame and save as JSONL
train_df = pd.DataFrame(train_prompts)
test_df = pd.DataFrame(test_prompts)

# Convert tensors to Python-native types (int or list) before saving
def tensor_to_json_compatible(record):
    return {key: (value.item() if isinstance(value, torch.Tensor) else value)
            for key, value in record.items()}

with open('mnist_train_prompts.jsonl', 'w') as f:
    for record in train_prompts:
        json_compatible_record = tensor_to_json_compatible(record)
        f.write(ujson.dumps(json_compatible_record) + '\n')

with open('mnist_test_prompts.jsonl', 'w') as f:
    for record in test_prompts:
        json_compatible_record = tensor_to_json_compatible(record)
        f.write(ujson.dumps(json_compatible_record) + '\n')