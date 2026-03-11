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

from aggregators import get_aggregator, get_torchjd_aggregator

try:
    from torchjd.autojac import backward as tj_backward
except Exception:
    tj_backward = None

try:
    from torchjd.autogram import Engine as TJEngine
except Exception:
    TJEngine = None

try:
    import torchjd.aggregation as tjagg
except Exception:
    tjagg = None


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
    health_stats: Dict[str, int]


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def runtrace_to_dict(trace: RunTrace) -> Dict:
    return {
        "seed": trace.seed,
        "best_lr": trace.best_lr,
        "train_loss_per_iter": trace.train_loss_per_iter,
        "sim_to_sgd_per_iter": trace.sim_to_sgd_per_iter,
        "health_stats": trace.health_stats,
    }


def runtrace_from_dict(data: Dict) -> RunTrace:
    return RunTrace(
        seed=int(data["seed"]),
        best_lr=float(data["best_lr"]),
        train_loss_per_iter=list(data["train_loss_per_iter"]),
        sim_to_sgd_per_iter=list(data["sim_to_sgd_per_iter"]),
        health_stats=dict(
            data.get(
                "health_stats",
                {
                    "total_batches": 0,
                    "skipped_nonfinite_loss_batches": 0,
                    "sanitized_j_batches": 0,
                    "sanitized_agg_batches": 0,
                },
            )
        ),
    )


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def load_existing_results(path: str) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def build_completed_cache(existing_results: Dict) -> Tuple[Dict[str, Dict[int, RunTrace]], Dict[str, Dict]]:
    completed: Dict[str, Dict[int, RunTrace]] = {}
    lr_trials_cache: Dict[str, Dict] = {}
    for agg_name, data in existing_results.items():
        runs = data.get("runs", [])
        by_seed: Dict[int, RunTrace] = {}
        if isinstance(runs, list):
            for item in runs:
                try:
                    trace = runtrace_from_dict(item)
                    by_seed[int(trace.seed)] = trace
                except (KeyError, TypeError, ValueError):
                    continue
        if by_seed:
            completed[agg_name] = by_seed
        lr_trials_cache[agg_name] = data.get("lr_trials", {})
    return completed, lr_trials_cache


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
    backend: str = "manual",
    agg_kwargs: Dict = None,
    show_bar: bool = False,
) -> Tuple[List[float], List[float], Dict[str, int]]:
    model.train()
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    params = [p for p in model.parameters() if p.requires_grad]

    loss_trace: List[float] = []
    sim_trace: List[float] = []
    bar_disable = not show_bar
    stats = {
        "total_batches": 0,
        "skipped_nonfinite_loss_batches": 0,
        "sanitized_j_batches": 0,
        "sanitized_agg_batches": 0,
    }

    backend = str(backend).lower()
    engine = None
    upgrad_weighting = None
    agg_kwargs = agg_kwargs or {}
    if backend == "autogram":
        if TJEngine is None or tjagg is None:
            raise ImportError("torchjd is required for autogram backend.")
        engine = TJEngine(model, batch_dim=0)
        upgrad_weighting = getattr(tjagg, "UPGradWeighting", None)
        if upgrad_weighting is None:
            raise ValueError("torchjd.aggregation.UPGradWeighting is unavailable.")
        epsilon = float(agg_kwargs.get("epsilon", 1e-8))
        upgrad_weighting = upgrad_weighting(epsilon=epsilon)

    for _ in range(num_epochs):
        for x, y in tqdm(loader, desc="Train", leave=False, disable=bar_disable):
            stats["total_batches"] += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            per_sample_losses = criterion(logits, y)  # [batch]
            if not torch.isfinite(per_sample_losses).all():
                # Skip unstable batch to avoid propagating NaN/Inf into Jacobian rows.
                stats["skipped_nonfinite_loss_batches"] += 1
                continue

            if backend == "manual":
                jac_rows = []
                for i in range(per_sample_losses.shape[0]):
                    optimizer.zero_grad(set_to_none=True)
                    per_sample_losses[i].backward(
                        retain_graph=(i + 1 < per_sample_losses.shape[0])
                    )
                    jac_rows.append(flatten_grads(params))

                J = torch.stack(jac_rows, dim=0)  # [m, n]
                if not torch.isfinite(J).all():
                    J = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
                    stats["sanitized_j_batches"] += 1
                agg = aggregator([row for row in J])
                if not torch.isfinite(agg).all():
                    agg = torch.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0)
                    stats["sanitized_agg_batches"] += 1
                mean_grad = J.mean(dim=0)
                sim = torch.nn.functional.cosine_similarity(
                    agg.unsqueeze(0), mean_grad.unsqueeze(0), dim=1, eps=1e-12
                ).item()

                optimizer.zero_grad(set_to_none=True)
                assign_flat_grad(params, agg)
                optimizer.step()
            else:
                if backend == "autogram":
                    gramian = engine.compute_gramian(per_sample_losses)
                    if not torch.isfinite(gramian).all():
                        gramian = torch.nan_to_num(gramian, nan=0.0, posinf=0.0, neginf=0.0)
                        stats["sanitized_j_batches"] += 1
                    weights = upgrad_weighting(gramian)
                    if not torch.isfinite(weights).all():
                        weights = torch.ones_like(weights) / weights.numel()
                        stats["sanitized_agg_batches"] += 1
                    optimizer.zero_grad(set_to_none=True)
                    per_sample_losses.backward(weights)
                    agg_grad = flatten_grads(params)
                    mean_grad = None
                else:
                    if tj_backward is None:
                        raise ImportError("torchjd is required for autojac backend.")
                    try:
                        optimizer.zero_grad(set_to_none=True)
                        tj_backward(per_sample_losses, inputs=params)
                        jac_blocks = []
                        for p in params:
                            if not hasattr(p, "jac") or p.jac is None:
                                raise RuntimeError("torchjd autojac did not populate .jac")
                            jac_blocks.append(p.jac.reshape(p.jac.size(0), -1))
                            p.jac = None
                        J = torch.cat(jac_blocks, dim=1)  # [m, n]
                        if not torch.isfinite(J).all():
                            J = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
                            stats["sanitized_j_batches"] += 1
                        mean_grad = J.mean(dim=0)
                        agg_grad = aggregator(J)
                    except Exception as e:
                        print(
                            f"torchjd autojac failed ({type(e).__name__}): {e}. Using mean."
                        )
                        optimizer.zero_grad(set_to_none=True)
                        per_sample_losses.mean().backward()
                        agg_grad = flatten_grads(params)
                        mean_grad = agg_grad

                if not torch.isfinite(agg_grad).all():
                    agg_grad = torch.nan_to_num(agg_grad, nan=0.0, posinf=0.0, neginf=0.0)
                    stats["sanitized_agg_batches"] += 1

                if mean_grad is None:
                    optimizer.zero_grad(set_to_none=True)
                    per_sample_losses.mean().backward(retain_graph=False)
                    mean_grad = flatten_grads(params)

                sim = torch.nn.functional.cosine_similarity(
                    agg_grad.unsqueeze(0), mean_grad.unsqueeze(0), dim=1, eps=1e-12
                ).item()
                assign_flat_grad(params, agg_grad)
                optimizer.step()

            loss_trace.append(per_sample_losses.mean().item())
            sim_trace.append(sim)

    return loss_trace, sim_trace, stats


def auc_of_loss(loss_trace: List[float]) -> float:
    if len(loss_trace) <= 1:
        return float(loss_trace[0]) if loss_trace else float("inf")
    x = np.arange(len(loss_trace), dtype=float)
    y = np.array(loss_trace, dtype=float)
    # NumPy 2.4 removed np.trapz in some builds; use trapezoid when available.
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x=x))
    return float(np.trapz(y, x=x))


def select_lr(
    base_model_state: Dict,
    loader: DataLoader,
    agg_name: str,
    agg_type: str,
    agg_kwargs: Dict,
    cfg: Dict,
    device: str,
    show_bar: bool,
    backend: str,
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
        aggregator = get_torchjd_aggregator(agg_type, **agg_kwargs) if backend != "manual" else get_aggregator(agg_type, **agg_kwargs)
        losses, _, _ = train_one_run(
            model=model,
            loader=loader,
            aggregator=aggregator,
            device=device,
            num_epochs=trial_epochs,
            lr=lr,
            momentum=float(cfg["training"]["momentum"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            backend=backend,
            agg_kwargs=agg_kwargs,
            show_bar=show_bar,
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
    if len(traces) == 0:
        raise ValueError("No run traces to aggregate.")

    loss_lengths = [len(t.train_loss_per_iter) for t in traces]
    sim_lengths = [len(t.sim_to_sgd_per_iter) for t in traces]
    common_len = min(min(loss_lengths), min(sim_lengths))
    if common_len <= 0:
        raise ValueError(
            f"Cannot aggregate empty traces. loss_lengths={loss_lengths}, sim_lengths={sim_lengths}"
        )
    if len(set(loss_lengths)) > 1 or len(set(sim_lengths)) > 1:
        print(
            f"[aggregate_runs] variable trace lengths detected, truncating to common_len={common_len}. "
            f"loss_lengths={loss_lengths}, sim_lengths={sim_lengths}"
        )

    loss_mat = np.array(
        [t.train_loss_per_iter[:common_len] for t in traces], dtype=float
    )
    sim_mat = np.array(
        [t.sim_to_sgd_per_iter[:common_len] for t in traces], dtype=float
    )
    n = loss_mat.shape[0]
    denom = max(1, int(math.sqrt(n)))

    return {
        "runs": [
            {
                "seed": t.seed,
                "best_lr": t.best_lr,
                "train_loss_per_iter": t.train_loss_per_iter,
                "sim_to_sgd_per_iter": t.sim_to_sgd_per_iter,
                "health_stats": t.health_stats,
            }
            for t in traces
        ],
        "train_loss_mean": loss_mat.mean(axis=0).tolist(),
        "train_loss_sem": (loss_mat.std(axis=0, ddof=0) / denom).tolist(),
        "sim_to_sgd_mean": sim_mat.mean(axis=0).tolist(),
        "sim_to_sgd_sem": (sim_mat.std(axis=0, ddof=0) / denom).tolist(),
        "health_summary": {
            "total_batches": int(sum(t.health_stats.get("total_batches", 0) for t in traces)),
            "skipped_nonfinite_loss_batches": int(
                sum(t.health_stats.get("skipped_nonfinite_loss_batches", 0) for t in traces)
            ),
            "sanitized_j_batches": int(
                sum(t.health_stats.get("sanitized_j_batches", 0) for t in traces)
            ),
            "sanitized_agg_batches": int(
                sum(t.health_stats.get("sanitized_agg_batches", 0) for t in traces)
            ),
        },
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
    backend = str(cfg.get("training", {}).get("iwrm_backend", "manual")).lower()

    os.makedirs(data_root, exist_ok=True)
    output_dir = cfg.get("logging", {}).get("log_dir", "./logs_paper")
    os.makedirs(output_dir, exist_ok=True)
    show_progress = bool(cfg.get("logging", {}).get("show_progress", False))
    resume_enabled = bool(cfg.get("logging", {}).get("resume", True))
    skip_completed = bool(cfg.get("logging", {}).get("skip_completed", True))
    partial_dir = os.path.join(output_dir, "partial")
    os.makedirs(partial_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "results.json")

    print(f"Using device: {device}")
    print(f"Seeds: {run_seeds}")
    print(f"Subset size: {subset_size}, batch size: {batch_size}, epochs: {num_epochs}")

    existing_results = load_existing_results(out_json) if resume_enabled else {}
    completed_cache, lr_trials_cache = build_completed_cache(existing_results)
    all_results = dict(existing_results)

    for agg_cfg in cfg.get("aggregators", []):
        agg_name = agg_cfg["name"]
        agg_type = agg_cfg["type"]
        agg_kwargs = {k: v for k, v in agg_cfg.items() if k not in ["name", "type"]}
        effective_backend = backend
        if backend == "autogram" and str(agg_type).lower() != "upgrad":
            print(
                f"[{agg_name}] autogram backend only supports UPGrad; "
                f"falling back to autojac."
            )
            effective_backend = "autojac"
        print("\n" + "=" * 70)
        print(f"Aggregator: {agg_name}")
        print("=" * 70)

        traces: List[RunTrace] = []
        lr_trials_per_seed: Dict[str, List[Dict]] = dict(lr_trials_cache.get(agg_name, {}))
        completed_seeds = set()

        if resume_enabled:
            for seed, trace in completed_cache.get(agg_name, {}).items():
                traces.append(trace)
                completed_seeds.add(int(seed))

        if skip_completed and len(completed_seeds) == len(run_seeds):
            print(f"[{agg_name}] already completed for all seeds, skipping.")
            continue

        for seed in run_seeds:
            if resume_enabled and int(seed) in completed_seeds:
                print(f"[{agg_name}] seed={seed} already in results.json, skip recompute.")
                continue
            partial_path = os.path.join(
                partial_dir, f"{_safe_name(agg_name)}_seed{int(seed)}.json"
            )
            if resume_enabled and os.path.exists(partial_path):
                with open(partial_path, "r", encoding="utf-8") as f:
                    partial_data = json.load(f)
                traces.append(runtrace_from_dict(partial_data["run_trace"]))
                lr_trials_per_seed[str(seed)] = partial_data.get("lr_trials", [])
                print(
                    f"[{agg_name}] seed={seed} loaded from partial checkpoint, "
                    f"skip recompute."
                )
                continue

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
                show_bar=show_progress,
                backend=effective_backend,
            )
            lr_trials_per_seed[str(seed)] = lr_trials

            model = PaperCIFARNet().to(device)
            model.load_state_dict(init_state)
            if effective_backend == "manual":
                aggregator = get_aggregator(agg_type, **agg_kwargs)
            else:
                aggregator = get_torchjd_aggregator(agg_type, **agg_kwargs)

            loss_trace, sim_trace, health_stats = train_one_run(
                model=model,
                loader=loader,
                aggregator=aggregator,
                device=device,
                num_epochs=num_epochs,
                lr=best_lr,
                momentum=momentum,
                weight_decay=weight_decay,
                backend=effective_backend,
                agg_kwargs=agg_kwargs,
                show_bar=show_progress,
            )
            traces.append(
                RunTrace(
                    seed=seed,
                    best_lr=best_lr,
                    train_loss_per_iter=loss_trace,
                    sim_to_sgd_per_iter=sim_trace,
                    health_stats=health_stats,
                )
            )
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "aggregator": agg_name,
                        "type": agg_type,
                        "seed": int(seed),
                        "run_trace": runtrace_to_dict(traces[-1]),
                        "lr_trials": lr_trials,
                    },
                    f,
                    indent=2,
                )
                f.flush()
            print(
                f"[{agg_name}] seed={seed} done. final_loss={loss_trace[-1]:.6f}, "
                f"final_sim={sim_trace[-1]:.6f}, lr={best_lr:.8f}, "
                f"skipped={health_stats.get('skipped_nonfinite_loss_batches', 0)}"
            )

        agg_result = aggregate_runs(traces)
        agg_result["lr_trials"] = lr_trials_per_seed
        all_results[agg_name] = agg_result
        # Persist merged results after each aggregator for safer long runs.
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    out_png = os.path.join(output_dir, "results.png")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    plot_results(all_results, out_png)

    print("\nDone.")
    print(f"Results JSON: {out_json}")
    print(f"Results plot: {out_png}")


if __name__ == "__main__":
    main()
