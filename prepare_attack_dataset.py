import os
from PIL import Image
import torch
from torchvision import transforms
import ujson

transform = transforms.Compose([transforms.ToTensor()])

attacked_images_dir = './attacked_images/'

def image_to_prompt(image, label):
    flattened_image = image.numpy().flatten()
    pixel_values = " ".join(map(str, flattened_image))
    prompt = f"Given image with pixels {pixel_values}. What is this digit?"
    return {"prompt": prompt, "label": label}

def tensor_to_json_compatible(record):
    return {key: (value.item() if isinstance(value, torch.Tensor) else value)
            for key, value in record.items()}

# Function to process the attacked images
def process_attacked_images(attacked_images_dir):
    epsilon_prompts = {}
    
    for ep_dir in os.listdir(attacked_images_dir):
        ep_path = os.path.join(attacked_images_dir, ep_dir)
        if os.path.isdir(ep_path):
            perturbation_value = ep_dir.split('_')[1]
            
            if perturbation_value not in epsilon_prompts:
                epsilon_prompts[perturbation_value] = []

            for img_name in os.listdir(ep_path):
                img_path = os.path.join(ep_path, img_name)
                if img_name.endswith('.png'):
                    # Load the image
                    image = Image.open(img_path).convert('L')  # Convert to grayscale
                    image_tensor = transform(image)
                    
                    label = int(img_name.split('_')[0])  # Modify based on how you want to extract the label
                    
                    # Generate the prompt for this image
                    prompt = image_to_prompt(image_tensor, label)
                    epsilon_prompts[perturbation_value].append(prompt)
    
    return epsilon_prompts

epsilon_prompts = process_attacked_images(attacked_images_dir)

# Save the prompts to a JSONL file for each epsilon value
for epsilon, prompts in epsilon_prompts.items():
    output_file = f'attacked_mnist_test_prompts_eps_{epsilon}.jsonl'
    with open(output_file, 'w') as f:
        for record in prompts:
            json_compatible_record = tensor_to_json_compatible(record)
            f.write(ujson.dumps(json_compatible_record) + '\n')
    
    print(f"Prompts for epsilon {epsilon} saved to '{output_file}'")
