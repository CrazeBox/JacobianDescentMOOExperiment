"""
Paper-aligned CIFAR-10 reproduction for:
"Jacobian Descent for Multi-Objective Optimization" (arXiv:2406.16232)

This script focuses on the IWRM + SSJD setup used in the paper:
- 1024 training examples per run
- batch size = number of rows in the sub-Jacobian (default 32)
- constant learning rate (selected per aggregator and per seed)
- repeated runs across multiple seeds with mean + SEM reporting
"""

import argparse
import json
import math
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from aggregators import get_aggregator


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def has_cifar10_data(root: str) -> bool:
    base_dir = os.path.join(root, "cifar-10-batches-py")
    required = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        "test_batch",
        "batches.meta",
    ]
    return os.path.isdir(base_dir) and all(
        os.path.exists(os.path.join(base_dir, name)) for name in required
    )


class PaperCIFARNet(nn.Module):
    """
    Architecture aligned with the paper's Appendix table for CIFAR-10.
    Uses grouped conv + ELU, ending with Linear(1024->128)->Linear(128->10).
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, groups=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, groups=32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, groups=64),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.ELU(inplace=True),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


@dataclass
class RunTrace:
    seed: int
    best_lr: float
    train_loss_per_iter: List[float]
    sim_to_sgd_per_iter: List[float]


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def build_lr_grid(lr_cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    min_exp = float(lr_cfg.get("coarse_min_exp", -5.0))
    max_exp = float(lr_cfg.get("coarse_max_exp", 2.0))
    coarse_num = int(lr_cfg.get("coarse_num", 22))
    refined_num = int(lr_cfg.get("refined_num", 50))
    coarse = np.logspace(min_exp, max_exp, num=coarse_num, base=10.0)
    return coarse, np.array([refined_num], dtype=np.int64)


def make_subset_loader(
    data_root: str,
    subset_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    augment: bool,
) -> DataLoader:
    transform_list = []
    if augment:
        transform_list.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    transform_list.extend([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    transform = transforms.Compose(transform_list)
    download = not has_cifar10_data(data_root)
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=download, transform=transform
    )
    subset_size = max(1, min(subset_size, len(dataset)))
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=rng)[:subset_size].tolist()
    subset = Subset(dataset, idx)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def flatten_grads(params: List[torch.nn.Parameter]) -> torch.Tensor:
    flat = []
    for p in params:
        if p.grad is None:
            flat.append(torch.zeros_like(p).reshape(-1))
        else:
            flat.append(p.grad.reshape(-1))
    return torch.cat(flat)


def assign_flat_grad(params: List[torch.nn.Parameter], grad: torch.Tensor) -> None:
    idx = 0
    for p in params:
        n = p.numel()
        p.grad = grad[idx : idx + n].view_as(p).clone()
        idx += n


def train_one_run(
    model: nn.Module,
    loader: DataLoader,
    aggregator,
    device: str,
    num_epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    show_bar: bool = False,
) -> Tuple[List[float], List[float]]:
    model.train()
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    params = [p for p in model.parameters() if p.requires_grad]

    loss_trace: List[float] = []
    sim_trace: List[float] = []
    bar_disable = not show_bar

    for _ in range(num_epochs):
        for x, y in tqdm(loader, desc="Train", leave=False, disable=bar_disable):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            per_sample_losses = criterion(logits, y)  # [batch]

            jac_rows = []
            for i in range(per_sample_losses.shape[0]):
                optimizer.zero_grad(set_to_none=True)
                per_sample_losses[i].backward(retain_graph=(i + 1 < per_sample_losses.shape[0]))
                jac_rows.append(flatten_grads(params))

            J = torch.stack(jac_rows, dim=0)  # [m, n]
            agg = aggregator([row for row in J])
            mean_grad = J.mean(dim=0)
            sim = torch.nn.functional.cosine_similarity(
                agg.unsqueeze(0), mean_grad.unsqueeze(0), dim=1, eps=1e-12
            ).item()

            optimizer.zero_grad(set_to_none=True)
            assign_flat_grad(params, agg)
            optimizer.step()

            loss_trace.append(per_sample_losses.mean().item())
            sim_trace.append(sim)

    return loss_trace, sim_trace


def auc_of_loss(loss_trace: List[float]) -> float:
    if len(loss_trace) <= 1:
        return float(loss_trace[0]) if loss_trace else float("inf")
    x = np.arange(len(loss_trace), dtype=float)
    y = np.array(loss_trace, dtype=float)
    return float(np.trapz(y, x=x))


def select_lr(
    base_model_state: Dict,
    loader: DataLoader,
    agg_name: str,
    agg_type: str,
    agg_kwargs: Dict,
    cfg: Dict,
    device: str,
) -> Tuple[float, List[Dict]]:
    search_cfg = cfg.get("learning_rate_search", {})
    enabled = bool(search_cfg.get("enabled", True))
    if not enabled:
        return float(cfg["training"]["learning_rate"]), []

    coarse, refined_num_arr = build_lr_grid(search_cfg)
    refined_num = int(refined_num_arr[0])
    margin_exp = float(search_cfg.get("refine_margin_exp", 1.0 / 3.0))
    trial_epochs = int(search_cfg.get("trial_epochs", cfg["training"]["num_epochs"]))

    trial_records = []

    def run_trial(lr: float) -> float:
        model = PaperCIFARNet().to(device)
        model.load_state_dict(base_model_state)
        aggregator = get_aggregator(agg_type, **agg_kwargs)
        losses, _ = train_one_run(
            model=model,
            loader=loader,
            aggregator=aggregator,
            device=device,
            num_epochs=trial_epochs,
            lr=lr,
            momentum=float(cfg["training"]["momentum"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            show_bar=False,
        )
        score = auc_of_loss(losses)
        return score

    for lr in coarse:
        score = run_trial(float(lr))
        trial_records.append({"phase": "coarse", "lr": float(lr), "auc": score})

    sorted_coarse = sorted([r for r in trial_records if r["phase"] == "coarse"], key=lambda x: x["auc"])
    best_two = sorted_coarse[:2]
    lo = min(best_two[0]["lr"], best_two[1]["lr"]) * (10.0 ** (-margin_exp))
    hi = max(best_two[0]["lr"], best_two[1]["lr"]) * (10.0 ** (margin_exp))
    refined = np.logspace(math.log10(lo), math.log10(hi), num=refined_num, base=10.0)

    for lr in refined:
        score = run_trial(float(lr))
        trial_records.append({"phase": "refined", "lr": float(lr), "auc": score})

    best = min(trial_records, key=lambda x: x["auc"])
    print(f"[{agg_name}] selected lr={best['lr']:.8f} (AUC={best['auc']:.6f})")
    return float(best["lr"]), trial_records


def aggregate_runs(traces: List[RunTrace]) -> Dict:
    loss_mat = np.array([t.train_loss_per_iter for t in traces], dtype=float)
    sim_mat = np.array([t.sim_to_sgd_per_iter for t in traces], dtype=float)
    n = loss_mat.shape[0]
    denom = max(1, int(math.sqrt(n)))

    return {
        "runs": [
            {
                "seed": t.seed,
                "best_lr": t.best_lr,
                "train_loss_per_iter": t.train_loss_per_iter,
                "sim_to_sgd_per_iter": t.sim_to_sgd_per_iter,
            }
            for t in traces
        ],
        "train_loss_mean": loss_mat.mean(axis=0).tolist(),
        "train_loss_sem": (loss_mat.std(axis=0, ddof=0) / denom).tolist(),
        "sim_to_sgd_mean": sim_mat.mean(axis=0).tolist(),
        "sim_to_sgd_sem": (sim_mat.std(axis=0, ddof=0) / denom).tolist(),
    }


def plot_results(results: Dict, out_png: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, data in results.items():
        x = np.arange(len(data["train_loss_mean"]))
        y = np.array(data["train_loss_mean"])
        sem = np.array(data["train_loss_sem"])
        axes[0].plot(x, y, label=name, linewidth=2)
        axes[0].fill_between(x, y - sem, y + sem, alpha=0.15)
    axes[0].set_title("Training Cross-Entropy over Iterations")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for name, data in results.items():
        x = np.arange(len(data["sim_to_sgd_mean"]))
        y = np.array(data["sim_to_sgd_mean"])
        sem = np.array(data["sim_to_sgd_sem"])
        axes[1].plot(x, y, label=name, linewidth=2)
        axes[1].fill_between(x, y - sem, y + sem, alpha=0.15)
    axes[1].set_title("Update Similarity to SGD over Iterations")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-aligned IWRM reproduction on CIFAR-10.")
    parser.add_argument("--config", type=str, default="config_paper_iwrm.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.setdefault("experiment", {})
        cfg["experiment"]["device"] = args.device

    device = cfg.get("experiment", {}).get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    run_seeds = [int(s) for s in cfg.get("experiment", {}).get("run_seeds", [0, 1, 2, 3, 4, 5, 6, 7])]
    data_root = cfg.get("dataset", {}).get("data_root", "./data")
    subset_size = int(cfg.get("dataset", {}).get("train_subset_size", 1024))
    batch_size = int(cfg.get("training", {}).get("batch_size", 32))
    num_workers = int(cfg.get("training", {}).get("num_workers", 2))
    num_epochs = int(cfg.get("training", {}).get("num_epochs", 20))
    momentum = float(cfg.get("training", {}).get("momentum", 0.0))
    weight_decay = float(cfg.get("training", {}).get("weight_decay", 0.0))
    augment = bool(cfg.get("dataset", {}).get("augment", False))

    os.makedirs(data_root, exist_ok=True)
    output_dir = cfg.get("logging", {}).get("log_dir", "./logs_paper")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Seeds: {run_seeds}")
    print(f"Subset size: {subset_size}, batch size: {batch_size}, epochs: {num_epochs}")

    all_results = {}

    for agg_cfg in cfg.get("aggregators", []):
        agg_name = agg_cfg["name"]
        agg_type = agg_cfg["type"]
        agg_kwargs = {k: v for k, v in agg_cfg.items() if k not in ["name", "type"]}
        print("\n" + "=" * 70)
        print(f"Aggregator: {agg_name}")
        print("=" * 70)

        traces: List[RunTrace] = []
        lr_trials_per_seed: Dict[str, List[Dict]] = {}

        for seed in run_seeds:
            set_seed(seed)
            loader = make_subset_loader(
                data_root=data_root,
                subset_size=subset_size,
                batch_size=batch_size,
                num_workers=num_workers,
                seed=seed,
                augment=augment,
            )

            init_model = PaperCIFARNet().to(device)
            init_state = deepcopy(init_model.state_dict())

            best_lr, lr_trials = select_lr(
                base_model_state=init_state,
                loader=loader,
                agg_name=agg_name,
                agg_type=agg_type,
                agg_kwargs=agg_kwargs,
                cfg=cfg,
                device=device,
            )
            lr_trials_per_seed[str(seed)] = lr_trials

            model = PaperCIFARNet().to(device)
            model.load_state_dict(init_state)
            aggregator = get_aggregator(agg_type, **agg_kwargs)

            loss_trace, sim_trace = train_one_run(
                model=model,
                loader=loader,
                aggregator=aggregator,
                device=device,
                num_epochs=num_epochs,
                lr=best_lr,
                momentum=momentum,
                weight_decay=weight_decay,
                show_bar=False,
            )
            traces.append(
                RunTrace(
                    seed=seed,
                    best_lr=best_lr,
                    train_loss_per_iter=loss_trace,
                    sim_to_sgd_per_iter=sim_trace,
                )
            )
            print(
                f"[{agg_name}] seed={seed} done. final_loss={loss_trace[-1]:.6f}, "
                f"final_sim={sim_trace[-1]:.6f}, lr={best_lr:.8f}"
            )

        agg_result = aggregate_runs(traces)
        agg_result["lr_trials"] = lr_trials_per_seed
        all_results[agg_name] = agg_result

    out_json = os.path.join(output_dir, "results.json")
    out_png = os.path.join(output_dir, "results.png")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    plot_results(all_results, out_png)

    print("\nDone.")
    print(f"Results JSON: {out_json}")
    print(f"Results plot: {out_png}")


if __name__ == "__main__":
    main()

