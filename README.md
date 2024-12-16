# Lift Filtering

## Description
This repository implements synaptic filtering techniques to evaluate and improve neural network robustness under adversarial attacks. The project explores how neural models perform under stress from both internal filtering and external adversarial inputs.

## Features
- Dataset preparation for adversarial attack experiments.
- Adversarial attack techniques: PGD, FGSM.
- Synaptic filtering to analyze model resilience.
- Evaluation and training scripts.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mzeeshankaramat/lift_filtering.git
   cd lift_filtering```
## Usage
1. Prepare datasets:
   ```
   python prepare_dataset.py
   ```
2. Generate adversarial datasets:
   ```
   python prepare_attack_dataset.py
   ```
3. Train the model:
   ```
   python training.py
   ```
4. Run inference with attacks:
   ```
   python attacked_inference.py
   ```
5. python synaptic_filtering.py:
   ```
   python synaptic_filtering.py
   ```
