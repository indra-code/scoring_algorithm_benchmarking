import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

def load_json(path):
    with open(path) as f:
        return json.load(f)

def percentage_to_stars(percentage):
    """Convert 0-100 percentage to 0.5-5 star rating using adjusted bins"""
    if percentage <= 10:
        return 0.5
    elif percentage <= 20:
        return 1.0
    elif percentage <= 30:
        return 1.5
    elif percentage <= 40:
        return 2.0
    elif percentage <= 50:
        return 2.5
    elif percentage <= 60:
        return 3.0
    elif percentage <= 70:
        return 3.5
    elif percentage <= 80:
        return 4.0
    elif percentage <= 87:
        return 4.5
    else:
        return 5.0

# Load data
model_scores = load_json("model_outputs.json")
human_stars = load_json("human_outputs.json")
common = sorted(set(model_scores.keys()) & set(human_stars.keys()))

model_raw = np.array([model_scores[k] for k in common])
human_stars_array = np.array([human_stars[k] for k in common])

print("=" * 80)
print("CALIBRATION OPTIMIZATION - EXTENSIVE SEARCH")
print("=" * 80)
print(f"Total images: {len(common)}\n")

calibrations = {}

# Power transformations (penalty curves)
for p in np.arange(1.0, 3.5, 0.1):
    calibrations[f"Power {p:.1f}"] = lambda x, p=p: (x / 100) ** p * 100

# Power + shift combinations
for p in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]:
    for shift in range(-15, 6, 1):
        name = f"Power {p:.1f} + shift {shift}"
        calibrations[name] = lambda x, p=p, s=shift: np.clip((x / 100) ** p * 100 + s, 0, 100)

# Piecewise linear (harsh on mid-range)
for threshold in [40, 50, 60, 70]:
    for low_mult in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for high_mult in [0.85, 0.9, 0.95, 1.0]:
            name = f"Piecewise t={threshold} l={low_mult:.1f} h={high_mult:.2f}"
            calibrations[name] = lambda x, t=threshold, lm=low_mult, hm=high_mult: (
                x * lm if x < t else (x - t) * hm + t * lm
            )

# Polynomial degree 2-5
for degree in [2, 3, 4, 5]:
    coeffs = np.polyfit(model_raw, human_stars_array, degree)
    poly = np.poly1d(coeffs)
    name = f"Polynomial deg={degree}"
    calibrations[name] = lambda x, poly=poly: np.clip(poly(x), 0.5, 5.0)

print(f"Testing {len(calibrations)} calibration functions...\n")

results = []

for name, func in calibrations.items():
    try:
        # Apply calibration to raw scores
        calibrated = np.array([func(s) for s in model_raw])
        
        # Convert to star ratings
        calibrated_stars = np.array([percentage_to_stars(s) for s in calibrated])
        
        # Calculate metrics
        exact_match = np.mean(calibrated_stars == human_stars_array)
        within_half = np.mean(np.abs(calibrated_stars - human_stars_array) <= 0.5)
        within_1 = np.mean(np.abs(calibrated_stars - human_stars_array) <= 1.0)
        mae = mean_absolute_error(human_stars_array, calibrated_stars)
        spearman, _ = spearmanr(human_stars_array, calibrated_stars)
        
        # Count unique predictions (diversity)
        unique_preds = len(np.unique(calibrated_stars))
        
        results.append({
            'name': name,
            'mae': mae,
            'exact': exact_match,
            'half': within_half,
            'one': within_1,
            'spearman': spearman,
            'unique': unique_preds,
            'func': func
        })
    except:
        pass

# Sort by MAE (lower is better)
results.sort(key=lambda x: x['mae'])

print(f"{'Method':<50} {'MAE':<8} {'Exact':<8} {'±0.5':<8} {'±1.0':<8} {'Spear':<8} {'Unique':<8}")
print("-" * 80)

# Show top 20 results
for r in results[:20]:
    print(f"{r['name']:<50} {r['mae']:<8.3f} {r['exact']:<8.1%} {r['half']:<8.1%} {r['one']:<8.1%} {r['spearman']:<8.3f} {r['unique']:<8}")

best = results[0]
print("\n" + "=" * 80)
print(f"BEST CALIBRATION: {best['name']}")
print("=" * 80)
print(f"  MAE: {best['mae']:.3f} stars")
print(f"  Exact Match: {best['exact']:.1%}")
print(f"  Within ±0.5 stars: {best['half']:.1%}")
print(f"  Within ±1.0 stars: {best['one']:.1%}")
print(f"  Spearman Correlation: {best['spearman']:.3f}")
print(f"  Unique Predictions: {best['unique']}/10 star levels")
print("=" * 80)

# Generate comparison plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Raw (no calibration)
raw_stars = np.array([percentage_to_stars(s) for s in model_raw])
axes[0, 0].scatter(model_raw, human_stars_array, alpha=0.6, s=40, c='blue')
x_line = np.linspace(0, 100, 200)
y_raw = [percentage_to_stars(x) for x in x_line]
axes[0, 0].plot(x_line, y_raw, 'r-', linewidth=2, label='Binning thresholds')
axes[0, 0].set_xlabel('Model Score (Raw %)')
axes[0, 0].set_ylabel('Star Rating')
axes[0, 0].set_title(f'Raw (No Calibration)\nMAE: {mean_absolute_error(human_stars_array, raw_stars):.3f} stars')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(0, 5.5)

# Plot 2: Best calibration
best_calibrated = np.array([best['func'](s) for s in model_raw])
best_stars = np.array([percentage_to_stars(s) for s in best_calibrated])
axes[0, 1].scatter(best_calibrated, human_stars_array, alpha=0.6, s=40, c='green')
y_best = [percentage_to_stars(x) for x in x_line]
axes[0, 1].plot(x_line, y_best, 'r-', linewidth=2, label='Binning thresholds')
axes[0, 1].set_xlabel('Model Score (Calibrated %)')
axes[0, 1].set_ylabel('Star Rating')
axes[0, 1].set_title(f'Best: {best["name"]}\nMAE: {best["mae"]:.3f} stars')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(0, 5.5)

# Plot 3: Calibration curve comparison
axes[1, 0].plot(x_line, x_line, 'k--', alpha=0.3, label='No calibration', linewidth=2)
best_curve = [best['func'](x) for x in x_line]
axes[1, 0].plot(x_line, best_curve, 'g-', linewidth=2, label=f'Best: {best["name"][:30]}')
axes[1, 0].set_xlabel('Raw Model Score (%)')
axes[1, 0].set_ylabel('Calibrated Score (%)')
axes[1, 0].set_title('Calibration Transformation Curve')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(0, 100)
axes[1, 0].set_ylim(0, 100)

# Plot 4: Error distribution
errors_raw = raw_stars - human_stars_array
errors_best = best_stars - human_stars_array
axes[1, 1].hist([errors_raw, errors_best], bins=np.arange(-5, 5.5, 0.5), 
                label=['Raw', 'Calibrated'], alpha=0.6)
axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Prediction Error (stars)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Error Distribution Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('calibration_optimization.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: calibration_optimization.png")

# Save best calibration details
with open('best_calibration_config.txt', 'w') as f:
    f.write(f"Best Calibration: {best['name']}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"MAE: {best['mae']:.3f} stars\n")
    f.write(f"Exact Match: {best['exact']:.1%}\n")
    f.write(f"Within ±0.5 stars: {best['half']:.1%}\n")
    f.write(f"Within ±1.0 stars: {best['one']:.1%}\n")
    f.write(f"Spearman Correlation: {best['spearman']:.3f}\n")
    f.write(f"Unique Predictions: {best['unique']}/10 star levels\n\n")
    
    # If it's a power function, extract parameters
    if "Power" in best['name'] and "shift" in best['name']:
        parts = best['name'].replace('Power ', '').split(' + shift ')
        power = parts[0]
        shift = parts[1]
        f.write(f"Implementation:\n")
        f.write(f"calibrated = (raw_score / 100) ** {power} * 100 + {shift}\n")
        f.write(f"calibrated = np.clip(calibrated, 0, 100)\n")

print("Saved: best_calibration_config.txt")
