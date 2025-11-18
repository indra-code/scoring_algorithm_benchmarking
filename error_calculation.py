import json
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score

def load_json(path):
    with open(path) as f:
        return json.load(f)

def score_to_stars(score):
    bins = [20, 40, 60, 80]
    idx = np.digitize([score], bins)[0]
    return int(idx + 1)

model_scores = load_json("model_outputs.json")
human_stars = load_json("human_outputs.json")

common = sorted(set(model_scores.keys()) & set(human_stars.keys()))

model_stars = [score_to_stars(model_scores[k]) for k in common]
human_stars_list = [human_stars[k] for k in common]

exact_acc = accuracy_score(human_stars_list, model_stars)
mae_stars = mean_absolute_error(human_stars_list, model_stars)
tolerance_acc = np.mean(np.abs(np.array(human_stars_list) - np.array(model_stars)) <= 1)
kappa = cohen_kappa_score(human_stars_list, model_stars, weights="quadratic")

print("Total images:", len(common))
print("Exact Star Accuracy:", exact_acc)
print("±1 Star Accuracy:", tolerance_acc)
print("Star MAE:", mae_stars)
print("Weighted Kappa:", kappa)
