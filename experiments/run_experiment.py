"""
Benchmark runner: trains one (task, dataset, activation, seed[, key_seed])
combination and appends a result row to a CSV.

Examples
--------
# Tabular, all activations, 3 seeds each:
python run_experiment.py --task tabular --dataset digits --seeds 0 1 2

# Vision, only cswish/kgelu, probing 3 different keys per model seed:
python run_experiment.py --task vision --dataset mnist \
    --activations cswish kgelu --seeds 0 --key-seeds 10 11 12

# Everything, quick tier:
python run_experiment.py --task tabular --dataset wine --seeds 0 1 2
python run_experiment.py --task tabular --dataset breast_cancer --seeds 0 1 2
python run_experiment.py --task vision --dataset mnist --seeds 0 1 2
python run_experiment.py --task vision --dataset fashion_mnist --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from activations import TASK_ACTIVATIONS, activation_factory, is_keyed
from datasets import load_tabular, load_vision
from models import MLP, SmallCNN

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_FIELDS = [
    "task", "dataset", "activation", "seed", "key_seed",
    "epochs", "final_train_loss", "final_val_loss", "best_val_acc",
    "final_val_acc", "epochs_to_90pct_of_best", "noise_val_acc", "wall_clock_s",
]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, device, noise_std: float = 0.0):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if noise_std > 0:
                x = x + noise_std * torch.randn_like(x)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * y.size(0)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def run_one(
    task: str,
    dataset: str,
    activation: str,
    seed: int,
    key_seed: Optional[int],
    epochs: int,
    lr: float,
    batch_size: int,
    hidden: int,
) -> dict:
    torch.manual_seed(seed)
    device = get_device()

    if task == "tabular":
        train_loader, val_loader, in_dim, n_classes = load_tabular(dataset, seed=seed, batch_size=batch_size)
        act_fac = activation_factory(activation, key_seed=key_seed)
        model = MLP(in_dim, n_classes, act_fac, hidden=hidden).to(device)
    elif task == "vision":
        train_loader, val_loader, shape, n_classes = load_vision(dataset, seed=seed, batch_size=batch_size)
        act_fac = activation_factory(activation, key_seed=key_seed)
        model = SmallCNN(shape, n_classes, act_fac).to(device)
    else:
        raise ValueError(f"Unknown task: {task!r}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    epochs_to_90pct = epochs  # default: never reached within budget
    val_acc_curve: List[float] = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * y.size(0)
            train_n += y.size(0)
        train_loss = train_loss_sum / train_n

        val_loss, val_acc = evaluate(model, val_loader, device)
        val_acc_curve.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    wall_clock = time.time() - t0

    # First epoch (1-indexed) at which val_acc reaches 90% of the eventual best.
    threshold = 0.9 * best_val_acc
    for i, acc in enumerate(val_acc_curve, start=1):
        if acc >= threshold:
            epochs_to_90pct = i
            break

    noise_loss, noise_acc = evaluate(model, val_loader, device, noise_std=0.1)

    return {
        "task": task,
        "dataset": dataset,
        "activation": activation,
        "seed": seed,
        "key_seed": key_seed if key_seed is not None else "",
        "epochs": epochs,
        "final_train_loss": round(train_loss, 5),
        "final_val_loss": round(val_loss, 5),
        "best_val_acc": round(best_val_acc, 5),
        "final_val_acc": round(val_acc_curve[-1], 5),
        "epochs_to_90pct_of_best": epochs_to_90pct,
        "noise_val_acc": round(noise_acc, 5),
        "wall_clock_s": round(wall_clock, 2),
    }


def append_csv(rows: List[dict], out_path: Path):
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["tabular", "vision"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--activations", nargs="*", default=None,
                         help="defaults to the task's standard activation set")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--key-seeds", nargs="*", type=int, default=[10, 11, 12],
                         help="only used for keyed activations (cswish/kgelu)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "benchmark.csv"))
    args = parser.parse_args()

    activations = args.activations or TASK_ACTIVATIONS[args.task]
    out_path = Path(args.out)
    device = get_device()
    print(f"[device] {device}")

    rows = []
    for activation in activations:
        if is_keyed(activation):
            for seed in args.seeds:
                for key_seed in args.key_seeds:
                    print(f"[run] {args.task}/{args.dataset} act={activation} seed={seed} key_seed={key_seed}")
                    rows.append(run_one(
                        args.task, args.dataset, activation, seed, key_seed,
                        args.epochs, args.lr, args.batch_size, args.hidden,
                    ))
        else:
            for seed in args.seeds:
                print(f"[run] {args.task}/{args.dataset} act={activation} seed={seed}")
                rows.append(run_one(
                    args.task, args.dataset, activation, seed, None,
                    args.epochs, args.lr, args.batch_size, args.hidden,
                ))

    append_csv(rows, out_path)
    print(f"[done] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
