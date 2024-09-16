import csv
import json
import os

# Paths
csv_file_path = 'data/processed_train.csv'  # Path to your CSV file
image_folder_path = '/scratch/m23csa016/aml/train'  # Folder where your images are stored
output_json_file = 'data/processed_train.json'  # Output JSON file

# Initialize list to store the dataset
dataset = []

# Open and read the CSV file
with open(csv_file_path, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    
    for row in reader:
        # Extract image filename from the image link
        image_url = row['image_link']
        image_filename = os.path.basename(image_url)  # Get just the filename, e.g., '61I9XdN6OFL.jpg'

        # Construct the image path in your train folder
        image_path = os.path.join(image_folder_path, image_filename)

        # Ensure the image exists in the train/ folder before adding to the dataset
        if os.path.exists(image_path):
            group_id = row['group_id']
            entity_name = row['entity_name']
            entity_value = row['entity_value']

            # Construct the conversation in ShareGPT format
            conversation = {
                "messages": [
                    {
                        "content": f"<image>What is the {entity_name}?",
                        "role": "user"
                    },
                    {
                        "content": f"{entity_value}",
                        "role": "assistant"
                    }
                ],
                "images": [
                    image_path
                ]
            }

            # Add the conversation to the dataset
            dataset.append(conversation)
        else:
            print(f"Image {image_filename} not found in {image_folder_path}, skipping.")

# Save the dataset to a JSON file
with open(output_json_file, 'w') as json_file:
    json.dump(dataset, json_file, indent=4)

print(f"Dataset successfully saved to {output_json_file}")