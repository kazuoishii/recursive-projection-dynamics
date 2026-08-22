import torch
import csv

LAYER_NAMES = ["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"]
TRAIN_DIR = "activations_final/train"
TEST_DIR = "activations_final/test"
OUT_CSV = "results/linear_probe_results.csv"
RIDGE = 1.0
NUM_CLASSES = 10

y_train = torch.load(f"{TRAIN_DIR}/labels.pt", map_location="cpu").long()
y_test = torch.load(f"{TEST_DIR}/labels.pt", map_location="cpu").long()

print("Final linear probe (ridge classifier)")
print("-------------------------------------")
print(f"Train samples : {len(y_train)}")
print(f"Test samples  : {len(y_test)}")
print(f"Ridge         : {RIDGE}")
print()

def standardize_train_test(X_train, X_test):
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return (X_train - mean) / std, (X_test - mean) / std

def one_hot(y):
    Y = torch.zeros(y.shape[0], NUM_CLASSES, dtype=torch.float64)
    Y[torch.arange(y.shape[0]), y] = 1.0
    return Y

def fit_ridge_probe(X, y, ridge=1.0):
    Y = one_hot(y)
    d = X.shape[1]
    XtX = X.T @ X
    XtY = X.T @ Y
    I = torch.eye(d, dtype=torch.float64)
    return torch.linalg.solve(XtX + ridge * I, XtY)

def accuracy(X, y, W):
    pred = (X @ W).argmax(dim=1)
    return (pred == y).double().mean().item()

rows = []
for name in LAYER_NAMES:
    X_train = torch.load(f"{TRAIN_DIR}/{name}.pt", map_location="cpu").double()
    X_test = torch.load(f"{TEST_DIR}/{name}.pt", map_location="cpu").double()
    if not torch.isfinite(X_train).all():
        raise ValueError(f"Non-finite TRAIN values: {name}")
    if not torch.isfinite(X_test).all():
        raise ValueError(f"Non-finite TEST values: {name}")
    X_train, X_test = standardize_train_test(X_train, X_test)
    W = fit_ridge_probe(X_train, y_train, ridge=RIDGE)
    train_acc = accuracy(X_train, y_train, W)
    test_acc = accuracy(X_test, y_test, W)
    rows.append({
        "layer": name,
        "feature_dimension": X_train.shape[1],
        "train_accuracy": train_acc,
        "test_accuracy": test_acc
    })
    print(f"{name:8s} features={X_train.shape[1]:4d} train={100*train_acc:6.2f}% test={100*test_acc:6.2f}%")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "layer", "feature_dimension", "train_accuracy", "test_accuracy"
    ])
    writer.writeheader()
    writer.writerows(rows)

print()
print(f"Saved: {OUT_CSV}")
