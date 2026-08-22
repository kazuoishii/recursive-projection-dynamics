import torch
import csv

LAYER_NAMES = [
    "conv1",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "avgpool",
]

ACT_DIR = "activations_final/test"
OUT_CSV = "results/effective_dimension_results.csv"

def effective_dimension(X):
    """
    Participation-ratio effective dimensionality:

        d_eff = (sum lambda_i)^2 / sum lambda_i^2
    """

    X = X.double()
    X = X - X.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(X)
    eigenvalues = s ** 2
    numerator = eigenvalues.sum() ** 2
    denominator = (eigenvalues ** 2).sum()
    return (numerator / denominator).item()

rows = []

print("Final effective dimensionality")
print("------------------------------")

for name in LAYER_NAMES:
    X = torch.load(f"{ACT_DIR}/{name}.pt", map_location="cpu")
    d_eff = effective_dimension(X)
    feature_dim = X.shape[1]
    normalized = d_eff / feature_dim
    rows.append({
        "layer": name,
        "feature_dimension": feature_dim,
        "effective_dimension": d_eff,
        "normalized_effective_dimension": normalized
    })
    print(f"{name:8s} features={feature_dim:4d} d_eff={d_eff:8.3f} normalized={normalized:.4f}")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "layer", "feature_dimension", "effective_dimension", "normalized_effective_dimension"
    ])
    writer.writeheader()
    writer.writerows(rows)

print()
print(f"Saved: {OUT_CSV}")
