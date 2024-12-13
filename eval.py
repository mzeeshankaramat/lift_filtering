import json
import re

from utils import calculate_accuracy

responses_file = 'generated_responses.json'
accuracy = calculate_accuracy(responses_file)
print(f'Accuracy: {accuracy}%')