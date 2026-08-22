import csv
import matplotlib.pyplot as plt

CKA_FILE = "results/cka_results.csv"
ED_FILE = "results/effective_dimension_results.csv"
PROBE_FILE = "results/linear_probe_results.csv"
OUT_PNG = "figures/representation_dynamics.png"
OUT_PDF = "figures/representation_dynamics.pdf"
layers = ["conv1", "layer1", "layer2", "layer3", "layer4"]

ed = {}
with open(ED_FILE, newline="") as f:
    for row in csv.DictReader(f):
        if row["layer"] in layers:
            ed[row["layer"]] = float(row["effective_dimension"])

probe = {}
with open(PROBE_FILE, newline="") as f:
    for row in csv.DictReader(f):
        if row["layer"] in layers:
            probe[row["layer"]] = float(row["test_accuracy"]) * 100.0

cka_transitions, cka_delta = [], []
with open(CKA_FILE, newline="") as f:
    for row in csv.DictReader(f):
        source, target = row["layer_from"], row["layer_to"]
        if source in layers and target in layers:
            cka_transitions.append(f"{source}\n→\n{target}")
            cka_delta.append(float(row["geometric_change"]))

if len(cka_delta) != 4:
    raise ValueError(f"Expected 4 CKA transitions, found {len(cka_delta)}")
for name in layers:
    if name not in ed or name not in probe:
        raise ValueError(f"Missing result: {name}")

print("Final figure values")
print("-------------------")
for i, name in enumerate(layers):
    delta_text = "-" if i == 0 else f"{cka_delta[i-1]:.4f}"
    print(f"{name:8s} Delta={delta_text:>6s} d_eff={ed[name]:7.3f} probe={probe[name]:6.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
fig.suptitle("Layer-wise Representation Dynamics in CIFAR-10 ResNet18", fontsize=16, y=0.98)

ax = axes[0]
x_cka = list(range(len(cka_delta)))
ax.plot(x_cka, cka_delta, marker="o", linewidth=2)
ax.set_xticks(x_cka)
ax.set_xticklabels(cka_transitions)
ax.set_ylabel(r"Geometric change $1-\mathrm{CKA}$")
ax.set_title("(a) Representation geometry", pad=12)
ax.set_ylim(0.12, 0.445)
ax.grid(alpha=0.3)
max_i = max(range(len(cka_delta)), key=lambda i: cka_delta[i])
ax.annotate(f"{cka_delta[max_i]:.3f}", xy=(max_i, cka_delta[max_i]), xytext=(-3, 8), textcoords="offset points", ha="center", va="bottom")

ax = axes[1]
x = list(range(len(layers)))
ed_values = [ed[name] for name in layers]
ax.plot(x, ed_values, marker="o", linewidth=2)
ax.set_xticks(x)
ax.set_xticklabels(layers, rotation=45, ha="right")
ax.set_ylabel("Effective dimension")
ax.set_title("(b) Effective dimensionality", pad=12)
ax.set_ylim(0, 22.5)
ax.grid(alpha=0.3)
i = layers.index("layer3")
ax.annotate(f"{ed['layer3']:.2f}", xy=(i, ed["layer3"]), xytext=(0, -16), textcoords="offset points", ha="center", va="top")
i = layers.index("layer4")
ax.annotate(f"{ed['layer4']:.2f}", xy=(i, ed["layer4"]), xytext=(-2, 8), textcoords="offset points", ha="center", va="bottom")

ax = axes[2]
probe_values = [probe[name] for name in layers]
ax.plot(x, probe_values, marker="o", linewidth=2)
ax.set_xticks(x)
ax.set_xticklabels(layers, rotation=45, ha="right")
ax.set_ylim(0, 108)
ax.set_ylabel("Linear-probe test accuracy (%)")
ax.set_title("(c) Task-relevant information", pad=12)
ax.grid(alpha=0.3)
i = layers.index("layer3")
ax.annotate(f"{probe['layer3']:.2f}%", xy=(i, probe["layer3"]), xytext=(0, 9), textcoords="offset points", ha="center", va="bottom")
i = layers.index("layer4")
ax.annotate(f"{probe['layer4']:.2f}%", xy=(i, probe["layer4"]), xytext=(0, -16), textcoords="offset points", ha="center", va="top")

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.subplots_adjust(top=0.78, wspace=0.22)
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print()
print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")
