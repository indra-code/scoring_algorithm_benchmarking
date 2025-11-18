import json
import os

json_path = "model_outputs.json"
images_dir = "images"

# 1. Load filenames from JSON
with open(json_path, "r") as f:
    data = json.load(f)

valid_filenames = set(data.keys())   

for fname in os.listdir(images_dir):
    file_path = os.path.join(images_dir, fname)

    if not os.path.isfile(file_path):
        continue

    if fname not in valid_filenames:
        print(f"Removing: {fname}")
        os.remove(file_path)

print("Done. Kept only JSON filenames.")
