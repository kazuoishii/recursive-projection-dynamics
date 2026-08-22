import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

LAYER_NAMES = ["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"]


def make_cifar_resnet18():
    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def main():
    p = argparse.ArgumentParser(description="Extract layer representations from the trained CIFAR-10 ResNet-18.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model", default="resnet18_final.pt")
    p.add_argument("--output-dir", default="activations_final")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--download", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=args.download, transform=transform)
    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=args.download, transform=transform)
    loaders = {
        "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0),
    }

    model = make_cifar_resnet18()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.to(device).eval()

    def extract(loader, split_name):
        split_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        activation_buffer = {name: [] for name in LAYER_NAMES}
        labels_buffer = []

        def make_hook(name):
            def hook(_module, _input, output):
                x = output.detach()
                if x.ndim == 4:
                    x = x.mean(dim=(2, 3))
                activation_buffer[name].append(x.reshape(x.size(0), -1).cpu())
            return hook

        handles = [getattr(model, name).register_forward_hook(make_hook(name)) for name in LAYER_NAMES]
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                _ = model(images.to(device))
                labels_buffer.append(labels.cpu())
                if (batch_idx + 1) % 50 == 0:
                    print(f"{split_name}: {(batch_idx + 1) * args.batch_size} samples processed")
        for handle in handles:
            handle.remove()

        labels = torch.cat(labels_buffer, dim=0)
        torch.save(labels, os.path.join(split_dir, "labels.pt"))
        for name in LAYER_NAMES:
            X = torch.cat(activation_buffer[name], dim=0)
            torch.save(X, os.path.join(split_dir, f"{name}.pt"))
            print(f"{split_name}/{name:8s} shape={tuple(X.shape)}")

    extract(loaders["train"], "train")
    extract(loaders["test"], "test")
    print(f"Saved under: {args.output_dir}/")


if __name__ == "__main__":
    main()
