import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import threading

load_dotenv()
API_ENDPOINT = os.getenv("API_ENDPOINT")
DIRECTORY = "images"
IMAGE_EXTENSIONS = ('.jpg','.jpeg','.png','.gif','.bmp')
OUTPUT_FILE = "model_outputs.json"

# Thread-safe lock for writing to JSON
file_lock = threading.Lock()
def send_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    try:
        with open(file_path,'rb') as f:
            files = {"image":(os.path.basename(file_path),f)}
            data = {
                "user_rating": "0",
                "startLatitude": "12.9753",
                "startLongitude": "77.591",
                "bearings": "10"
            }
            response = requests.post(API_ENDPOINT,files = files,data = data)
            response.raise_for_status()
            filename = os.path.basename(file_path)
            print(f"Uploaded {filename} successfully.")
            json_data = response.json()
            score = json_data['Percentage']
            print("--------Response JSON:", json_data)
            print("Footpath score:", score)
            
            # Save immediately to JSON
            with file_lock:
                # Load existing data
                if os.path.exists(OUTPUT_FILE):
                    with open(OUTPUT_FILE, 'r') as f:
                        result_dict = json.load(f)
                else:
                    result_dict = {}
                
                # Add new result
                result_dict[filename] = score
                
                # Write back
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(result_dict, f, indent=4)
            
            return (score, filename)
    except Exception as e:
        print(f"An error occurred with {os.path.basename(file_path)}: {e}")
        return None
def main():
    files = [
        os.path.join(DIRECTORY, f)
        for f in os.listdir(DIRECTORY)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    print(f"Starting upload of {len(files)} files...")
    print(f"Results will be saved to {OUTPUT_FILE} as they complete.\n")

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(send_file, files))
    
    uploaded = sum(1 for r in results if r is not None)
    print(f"\nCompleted! Files successfully uploaded: {uploaded} out of {len(files)}")


if __name__ == "__main__":
    main()