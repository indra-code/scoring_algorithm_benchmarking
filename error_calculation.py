import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

def load_json(path):
    with open(path) as f:
        return json.load(f)

def stars_to_score(stars):
    """Convert 0.5-5 star rating to 0-100 scale"""
    return (stars - 0.5) * (100 / 4.5)  # Maps 0.5→0, 5→100

def calibrate_model_score(raw_score):
    """
    Apply calibration to match human perception.
    Power 1.2 with -14 shift - empirically optimized for half-star ratings.
    This pulls down scores to account for quality issues humans perceive.
    """
    # Power 1.2 + shift -14 (MAE: 0.440, Exact: 38.7%, ±1 star: 93.3%)
    calibrated = (raw_score / 100) ** 1.2 * 100 - 14
    return float(np.clip(calibrated, 0, 100))

def percentage_to_stars(percentage):
    """Convert 0-100 percentage to 0.5-5 star rating using bins (half-star increments)"""
    # Adjusted thresholds to better match real-world model outputs
    # Lower threshold for 5 stars since model rarely hits 90%+ due to road inclusion
    if percentage <= 10:  # 0-10 → 0.5 stars
        return 0.5
    elif percentage <= 20:  # 10-20 → 1 star
        return 1.0
    elif percentage <= 30:  # 20-30 → 1.5 stars
        return 1.5
    elif percentage <= 40:  # 30-40 → 2 stars
        return 2.0
    elif percentage <= 50:  # 40-50 → 2.5 stars
        return 2.5
    elif percentage <= 60:  # 50-60 → 3 stars
        return 3.0
    elif percentage <= 70:  # 60-70 → 3.5 stars
        return 3.5
    elif percentage <= 80:  # 70-80 → 4 stars
        return 4.0
    elif percentage <= 87:  # 80-87 → 4.5 stars (tighter for high quality)
        return 4.5
    else:  # 87-100 → 5 stars (lowered from 94.44 to account for road inclusion)
        return 5.0

model_scores = load_json("model_outputs.json")
human_stars = load_json("human_outputs.json")

common = sorted(set(model_scores.keys()) & set(human_stars.keys()))

# Get model predictions
model_raw = np.array([model_scores[k] for k in common])
model_calibrated = np.array([calibrate_model_score(s) for s in model_raw])
model_stars = np.array([percentage_to_stars(s) for s in model_calibrated])

# Get human ratings
human_stars_array = np.array([human_stars[k] for k in common])

# Calculate classification metrics (star-based)
exact_match = np.mean(model_stars == human_stars_array)
mae_stars = mean_absolute_error(human_stars_array, model_stars)
within_half_star = np.mean(np.abs(model_stars - human_stars_array) <= 0.5)
within_1_star = np.mean(np.abs(model_stars - human_stars_array) <= 1.0)

# Spearman correlation on ordinal data (stars)
spearman_corr, spearman_p = spearmanr(human_stars_array, model_stars)

# Create confusion matrix for half-star ratings
star_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
cm = np.zeros((len(star_levels), len(star_levels)), dtype=int)

for h, m in zip(human_stars_array, model_stars):
    h_idx = star_levels.index(h)
    m_idx = star_levels.index(m)
    cm[h_idx][m_idx] += 1

print("=" * 70)
print("CLASSIFICATION METRICS (Half-Star Rating Evaluation)")
print("=" * 70)
print(f"Total images: {len(common)}")
print(f"\nAccuracy Metrics:")
print(f"  Exact Match Accuracy: {exact_match:.1%}")
print(f"  Within ±0.5 Stars: {within_half_star:.1%}")
print(f"  Within ±1.0 Stars: {within_1_star:.1%}")
print(f"  Mean Absolute Error: {mae_stars:.2f} stars")

print(f"\nCorrelation (Ordinal):")
print(f"  Spearman Correlation: {spearman_corr:.3f} (p={spearman_p:.4f})")

print(f"\n{'Confusion Matrix (Half-Star Ratings):'}")
print(f"{'':>8} Predicted Stars")
header = "Human" + "".join([f"{s:>6.1f}" for s in star_levels])
print(f"{header}")
print("-" * 70)
for i, (star_val, row) in enumerate(zip(star_levels, cm)):
    row_str = "".join([f"{val:>6}" for val in row])
    print(f"{star_val:>6.1f} {row_str}")

print("=" * 70)
