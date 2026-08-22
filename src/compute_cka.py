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
OUT_CSV = "results/cka_results.csv"

def center(X):
    return X - X.mean(dim=0, keepdim=True)

def linear_cka(X, Y):
    X = center(X.double())
    Y = center(Y.double())

    cross = X.T @ Y
    xx = X.T @ X
    yy = Y.T @ Y

    numerator = torch.norm(cross, p="fro") ** 2
    denominator = (
        torch.norm(xx, p="fro")
        * torch.norm(yy, p="fro")
    )

    return (numerator / denominator).item()

activations = {}

for name in LAYER_NAMES:
    path = f"{ACT_DIR}/{name}.pt"
    X = torch.load(path, map_location="cpu")
    activations[name] = X
    print(f"Loaded {name:8s} {tuple(X.shape)}")

print()
print("Final adjacent-layer CKA")
print("------------------------")

rows = []

for i in range(len(LAYER_NAMES) - 1):
    a = LAYER_NAMES[i]
    b = LAYER_NAMES[i + 1]

    cka = linear_cka(
        activations[a],
        activations[b]
    )

    delta = 1.0 - cka

    rows.append({
        "layer_from": a,
        "layer_to": b,
        "cka": cka,
        "geometric_change": delta
    })

    print(
        f"{a:8s} -> {b:8s} "
        f"CKA={cka:.4f} "
        f"Delta={delta:.4f}"
    )

with open(
    OUT_CSV,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "layer_from",
            "layer_to",
            "cka",
            "geometric_change"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print()
print(f"Saved: {OUT_CSV}")
