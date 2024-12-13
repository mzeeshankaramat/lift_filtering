import json
import re

def extract_number(generation):
    # Extract the first number from the response using regex
    match = re.search(r'\d+', generation)
    return int(match.group()) if match else None

def read_responses_and_labels(responses_file):
    # Read responses.json
    with open(responses_file, 'r') as file:
        data = json.load(file)

    # Extract responses and labels
    response_texts = [item['response'] for item in data]
    labels = [item['label'] for item in data]

    return response_texts, labels

def calculate_accuracy(responses_file):
    # Read responses and labels
    response_texts, labels = read_responses_and_labels(responses_file)

    # Ensure data alignment
    if len(response_texts) != len(labels):
        raise ValueError("Mismatch in number of responses and labels.")

    # Extract numbers from responses and compare with labels
    correct_matches = 0
    total_samples = len(labels)

    for response, label in zip(response_texts, labels):
        extracted_number = extract_number(response)
        print(f"extracted_number: {extracted_number}")
        print(f"label: {label}")
        if extracted_number == label:
            correct_matches += 1

    # Calculate accuracy
    accuracy = (correct_matches / total_samples) * 100
    return accuracy

