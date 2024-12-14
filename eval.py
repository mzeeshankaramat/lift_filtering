import json
import re
import argparse

from utils import calculate_accuracy

# Main function to handle argument parsing
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate accuracy from generated responses")
    parser.add_argument('--responses_file', type=str, required=True, help="Path to the responses file")
    
    # Parse the arguments
    args = parser.parse_args()

    # Calculate accuracy using the provided responses file
    accuracy = calculate_accuracy(args.responses_file)
    print(f'Accuracy: {accuracy}%')

if __name__ == "__main__":
    main()
