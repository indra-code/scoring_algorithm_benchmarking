# Footpath Scoring System

## Setup and Installation

1. **Install required packages:**
   ```powershell
   pip install -r requirements.txt
   ```

## Running the Application

### Step 1: Rate Images with Human Rating Tool

Run the Streamlit app to manually rate footpath images:

```powershell
python -m streamlit run human_rating.py
```

- Navigate through images using the Next/Previous buttons
- Rate each image from 1-5 stars
- Ratings are automatically saved when you navigate between images
- Results are saved to `human_outputs.json`

### Step 2: Calculate Errors

After rating all images, run the error calculation script:

```powershell
python error_calculation.py
```

This will compare human ratings with model predictions and calculate accuracy metrics.
