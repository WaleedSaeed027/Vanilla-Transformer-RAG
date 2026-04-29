import json
import random
import os

# Configuration
CATEGORIES = ['sports.json', 'home.json', 'electronics.json']
SAMPLES_PER_CAT = 10000
OUTPUT_FILE = 'sampled_dataset.json'

# Set seed for reproducibility so you get the same 30k reviews every time
random.seed(42)

def sample_data():
    sampled_data = []
    
    for file_path in CATEGORIES:
        print(f"Processing {file_path}...")
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found! Make sure it is in your current WSL directory.")
            continue
            
        cat_data = []
        
        # Stream the file line by line to keep your RAM usage low
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line.strip())
                    text = review.get('reviewText', '').strip()
                    rating = review.get('overall', None)
                    
                    # Only keep reviews that actually have text and a rating
                    if text and rating is not None:
                        cat_data.append({
                            'text': text,
                            'rating': rating
                        })
                except json.JSONDecodeError:
                    continue
        
        print(f"Found {len(cat_data)} valid reviews in {file_path}.")
        
        # Shuffle and pick exactly 10,000
        random.shuffle(cat_data)
        selected = cat_data[:SAMPLES_PER_CAT]
        
        # Add a category tag just in case you want to use it for debugging later
        for item in selected:
            item['category'] = file_path.split('.')[0]
            
        sampled_data.extend(selected)
        print(f"Successfully sampled {len(selected)} reviews from {file_path}.\n")

    # Shuffle the final combined 30k dataset so the categories are mixed during training
    random.shuffle(sampled_data)
    
    # Write everything to the final, clean JSON file
    print(f"Writing {len(sampled_data)} total samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=4)
        
    print("Done! You can now load sampled_dataset.json safely in your Jupyter Notebook.")

if __name__ == "__main__":
    sample_data()