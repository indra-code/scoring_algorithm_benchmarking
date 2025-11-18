import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

def load_json(path):
    with open(path) as f:
        return json.load(f)

def stars_to_score(stars):
    """Convert 1-5 star rating to 0-100 scale"""
    return (stars - 1) * 25  # 1→0, 2→25, 3→50, 4→75, 5→100

model_scores = load_json("model_outputs.json")
human_stars = load_json("human_outputs.json")

common = sorted(set(model_scores.keys()) & set(human_stars.keys()))

# Get raw model scores (0-100)
model_percentages = np.array([model_scores[k] for k in common])

# Convert human stars (1-5) to percentage scale (0-100)
human_percentages = np.array([stars_to_score(human_stars[k]) for k in common])

# Calculate metrics on 0-100 scale
mae = mean_absolute_error(human_percentages, model_percentages)
rmse = np.sqrt(mean_squared_error(human_percentages, model_percentages))
r2 = r2_score(human_percentages, model_percentages)
pearson_corr, pearson_p = pearsonr(human_percentages, model_percentages)
spearman_corr, spearman_p = spearmanr(human_percentages, model_percentages)

# Calculate percentage of predictions within tolerance ranges
within_10 = np.mean(np.abs(model_percentages - human_percentages) <= 10)
within_15 = np.mean(np.abs(model_percentages - human_percentages) <= 15)
within_25 = np.mean(np.abs(model_percentages - human_percentages) <= 25)

print("=" * 50)
print("PERFORMANCE METRICS (0-100 Scale)")
print("=" * 50)
print(f"Total images: {len(common)}")
print(f"\nError Metrics:")
print(f"  Mean Absolute Error (MAE): {mae:.2f} points")
print(f"  Root Mean Squared Error (RMSE): {rmse:.2f} points")
print(f"\nCorrelation:")
print(f"  Pearson Correlation: {pearson_corr:.3f} (p={pearson_p:.4f})")
print(f"  Spearman Correlation: {spearman_corr:.3f} (p={spearman_p:.4f})")
print(f"  R² Score: {r2:.3f}")
print(f"\nAccuracy within Tolerance:")
print(f"  Within ±10 points: {within_10:.1%}")
print(f"  Within ±15 points: {within_15:.1%}")
print(f"  Within ±25 points (1 star): {within_25:.1%}")
print("=" * 50)
