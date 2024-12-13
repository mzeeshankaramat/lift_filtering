import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,LlamaConfig, LlamaForCausalLM
import torch
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
import os
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import math

os.environ["WANDB_PROJECT"] = "lift_new"

# Load the JSONL data
def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return [{"input": item["prompt"], "output": str(item['label'])} for item in tqdm(data)]


class DigitDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = f"{item['input']} Answer: {item['output']}"
        
        # Tokenize the full sequence
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Create labels, masking prompt tokens with -100
        labels = tokens["input_ids"].clone()
        input_len = len(self.tokenizer(item['input'])["input_ids"])
        labels[:, :input_len] = -100

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }
        
# Load and preprocess the dataset
file_path = "mnist_train_prompts.jsonl"
data = load_data(file_path)

# print(data[0])


tokenizer = AutoTokenizer.from_pretrained('NousResearch/Meta-Llama-3.1-8B')

# Add padding token if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
train_dataset = DigitDataset(data, tokenizer)
    
# Resize token embeddings if new tokens are added
model = AutoModelForCausalLM.from_pretrained('NousResearch/Meta-Llama-3.1-8B')

lora_config = LoraConfig(r=4, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1)
model = get_peft_model(model, lora_config)

# model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir="./llama-finetuned",
    per_device_train_batch_size=4,  # Reduce batch size to prevent OOM
    gradient_accumulation_steps=16,  # Accumulate gradients over 16 steps
    num_train_epochs=8,
    save_steps=250,
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=1e-4,
    warmup_steps=500,
    weight_decay=0.01,
    bf16=False,  # Enable mixed precision if supported
    fp16=False,  # Enable mixed precision if supported
    report_to="wandb",
    push_to_hub=False,
    dataloader_num_workers=8,  # Parallelize data loading
    ddp_find_unused_parameters=False,  # Ensure efficient multi-GPU training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()