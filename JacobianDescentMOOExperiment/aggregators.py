"""
Multi-objective gradient aggregators for Jacobian Descent.
Implementation of aggregators from the paper.
"""

import torch
import numpy as np
from typing import List
import io
import contextlib
try:
    import cvxpy as cp
except ImportError:
    cp = None

try:
    import torchjd.aggregation as tjagg
except ImportError:
    tjagg = None


def _solve_problem_quietly(prob):
    """Solve CVXPY problem while suppressing noisy solver stdout/stderr."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            prob.solve(solver=cp.SCS, verbose=False, warm_start=True)
        except Exception:
            try:
                prob.solve(solver=cp.OSQP, verbose=False, polish=False, warm_start=True)
            except Exception:
                prob.solve(verbose=False)


def _stable_gram_matrix(J: torch.Tensor, epsilon: float) -> np.ndarray:
    """
    Build a numerically stable PSD Gram matrix for CVXPY:
    - enforce symmetry
    - move to float64
    - shift diagonal if tiny negative eigenvalues appear
    """
    m = J.size(0)
    G = torch.matmul(J, J.t())
    G = 0.5 * (G + G.t())
    G = G + float(epsilon) * torch.eye(m, device=G.device, dtype=G.dtype)
    G_np = G.detach().double().cpu().numpy()
    G_np = 0.5 * (G_np + G_np.T)
    try:
        eig_min = float(np.linalg.eigvalsh(G_np).min())
        if eig_min < 1e-10:
            G_np = G_np + (abs(eig_min) + 1e-8) * np.eye(m, dtype=np.float64)
    except Exception:
        # If eigendecomposition fails, still keep a conservative diagonal shift.
        G_np = G_np + 1e-6 * np.eye(m, dtype=np.float64)
    return G_np


class Aggregator:
    """Base class for multi-objective aggregators."""
    
    def __call__(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate multiple task gradients into a single update direction.
        
        Args:
            gradients: List of gradient tensors, one per task
            
        Returns:
            Aggregated gradient tensor
        """
        raise NotImplementedError


class MeanAggregator(Aggregator):
    """
    Simple mean aggregation baseline.
    Equivalent to standard multi-task learning with equal task weights.
    """
    
    def __call__(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(gradients).mean(dim=0)


class UPGradAggregator(Aggregator):
    """
    UPGrad (Unconflicting Projection Gradient) aggregator.
    From: "Jacobian Descent for Multi-Objective Optimization" (Quinton & Rey, 2024)
    
    Projects gradients to resolve conflicts while maintaining proportional influence.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = float(epsilon)
    
    def __call__(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        if len(gradients) == 1:
            return gradients[0]
        
        # Stack gradients into Jacobian matrix J [num_tasks, num_params]
        J = torch.stack(gradients)
        m = J.size(0)  # number of tasks
        
        # Compute Gram matrix G = J @ J^T
        G = torch.matmul(J, J.t())
        G = 0.5 * (G + G.t())
        G = G + self.epsilon * torch.eye(m, device=G.device)
        
        # Solve for optimal weights using QP formulation
        # min_w 0.5 * w^T G w - 1^T w
        # s.t. w >= 0
        
        try:
            if cp is None:
                # Fallback when cvxpy is unavailable: solve unconstrained system and clamp.
                e = torch.ones(m, dtype=J.dtype, device=J.device)
                weights = torch.linalg.lstsq(G, e).solution
                weights = torch.clamp(weights, min=0.0)
                if torch.sum(weights) <= self.epsilon:
                    weights = torch.ones(m, dtype=J.dtype, device=J.device) / m
                else:
                    weights = weights / torch.sum(weights)
            else:
                # Convert to stable PSD numpy matrix for CVXPY
                G_np = _stable_gram_matrix(J, self.epsilon)
                
                # Define optimization variable
                w = cp.Variable(m)
                
                # Define objective: minimize 0.5 * w^T G w - sum(w)
                objective = cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(G_np)) - cp.sum(w))
                
                # Constraints: w >= 0
                constraints = [w >= 0]
                
                # Solve QP
                prob = cp.Problem(objective, constraints)
                _solve_problem_quietly(prob)
                
                if w.value is not None:
                    weights = torch.tensor(w.value, dtype=J.dtype, device=J.device)
                    # Normalize weights
                    weights = weights / weights.sum()
                else:
                    # Fallback to uniform weights
                    weights = torch.ones(m, dtype=J.dtype, device=J.device) / m
        except Exception:
            weights = torch.ones(m, dtype=J.dtype, device=J.device) / m
        
        # Compute weighted combination
        aggregated = torch.matmul(weights, J)
        return aggregated


class MGDAAggregator(Aggregator):
    """
    Multiple Gradient Descent Algorithm (MGDA) aggregator.
    From: "Multiple-gradient descent algorithm (MGDA) for multiobjective optimization" (Désidéri, 2009)
    
    Finds the minimum-norm point in the convex hull of gradients.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = float(epsilon)
    
    def __call__(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        if len(gradients) == 1:
            return gradients[0]
        
        J = torch.stack(gradients)
        m = J.size(0)
        
        # Compute Gram matrix
        G = torch.matmul(J, J.t())
        G = 0.5 * (G + G.t())
        G = G + self.epsilon * torch.eye(m, device=G.device)
        
        try:
            if cp is None:
                # Fallback when cvxpy is unavailable.
                weights = torch.ones(m, dtype=J.dtype, device=J.device) / m
            else:
                # Solve min ||sum(w_i * grad_i)||^2 s.t. sum(w_i)=1, w_i>=0
                G_np = _stable_gram_matrix(J, self.epsilon)
                
                w = cp.Variable(m)
                
                # Objective: minimize w^T G w (norm of weighted sum)
                objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(G_np)))
                
                # Constraints
                constraints = [
                    w >= 0,
                    cp.sum(w) == 1
                ]
                
                prob = cp.Problem(objective, constraints)
                _solve_problem_quietly(prob)
                
                if w.value is not None:
                    weights = torch.tensor(w.value, dtype=J.dtype, device=J.device)
                else:
                    weights = torch.ones(m, dtype=J.dtype, device=J.device) / m
        except Exception:
            weights = torch.ones(m, dtype=J.dtype, device=J.device) / m
        
        aggregated = torch.matmul(weights, J)
        return aggregated


class CAGradAggregator(Aggregator):
    """
    Conflict-Averse Gradient (CAGrad) aggregator.
    From: "Conflict-Averse Gradient Descent for Multi-task Learning" (Liu et al., 2021)
    
    Adjusts gradients to reduce conflict while maintaining average direction.
    """
    
    def __init__(self, c: float = 0.5, epsilon: float = 1e-8):
        self.c = float(c)
        self.epsilon = float(epsilon)
    
    def __call__(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        if len(gradients) == 1:
            return gradients[0]
        
        J = torch.stack(gradients)
        if not torch.isfinite(J).all():
            # Fallback early if upstream gradients already contain NaN/Inf.
            return torch.nan_to_num(J.mean(dim=0), nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute average gradient
        avg_grad = J.mean(dim=0)
        if not torch.isfinite(avg_grad).all():
            return torch.nan_to_num(avg_grad, nan=0.0, posinf=0.0, neginf=0.0)
        denom = torch.clamp(avg_grad.square().sum(), min=self.epsilon)
        
        # Adjust individual gradients
        adjusted_grads = []
        for grad in gradients:
            # Compute dot product
            dot = torch.sum(grad * avg_grad)
            
            # If conflicting (dot < 0), project away from conflict
            if dot.item() < -self.epsilon:
                # Project grad onto avg_grad
                proj = (dot / denom) * avg_grad
                adjusted = grad - (1 + self.c) * proj
            else:
                adjusted = grad
            adjusted = torch.nan_to_num(adjusted, nan=0.0, posinf=0.0, neginf=0.0)
            
            adjusted_grads.append(adjusted)
        
        # Return mean of adjusted gradients
        aggregated = torch.stack(adjusted_grads).mean(dim=0)
        return aggregated


class PCGradAggregator(Aggregator):
    """
    Project Conflicting Gradients (PCGrad) aggregator.
    From: "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020)
    
    Projects gradients to remove conflicts pairwise.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = float(epsilon)
    
    def __call__(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        if len(gradients) == 1:
            return gradients[0]

        # Start with a copy of gradients and sanitize non-finite values.
        adjusted_grads = [
            torch.nan_to_num(g.clone(), nan=0.0, posinf=0.0, neginf=0.0) for g in gradients
        ]
        
        # Pairwise projection
        for i in range(len(adjusted_grads)):
            for j in range(len(adjusted_grads)):
                if i != j:
                    grad_i = adjusted_grads[i]
                    grad_j = adjusted_grads[j]
                    
                    # Compute dot product
                    dot = torch.sum(grad_i * grad_j)
                    
                    # If conflicting, project
                    if dot.item() < -self.epsilon:
                        # Project grad_i onto grad_j
                        denom = torch.clamp(grad_j.square().sum(), min=self.epsilon)
                        proj = (dot / denom) * grad_j
                        adjusted_grads[i] = grad_i - proj
                    adjusted_grads[i] = torch.nan_to_num(
                        adjusted_grads[i], nan=0.0, posinf=0.0, neginf=0.0
                    )
        
        # Return mean of adjusted gradients
        aggregated = torch.stack(adjusted_grads).mean(dim=0)
        return aggregated


class TorchJDAggregator(Aggregator):
    """Thin wrapper around torchjd aggregators when torchjd is available."""

    def __init__(self, class_name: str, **kwargs):
        if tjagg is None:
            raise ImportError(
                f"Aggregator '{class_name}' requires torchjd. "
                f"Install it via `pip install torchjd`."
            )
        cls = getattr(tjagg, class_name, None)
        if cls is None:
            raise ValueError(f"torchjd.aggregation has no class '{class_name}'.")
        self.inner = cls(**kwargs)

    def __call__(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        J = torch.stack(gradients)
        try:
            return self.inner(J)
        except Exception as e:
            # Fall back to mean to keep training running on ill-conditioned cases.
            print(f"torchjd aggregator failed ({type(e).__name__}): {e}. Using mean.")
            return J.mean(dim=0)


def get_aggregator(name: str, **kwargs) -> Aggregator:
    """
    Factory function to get aggregator by name.
    
    Args:
        name: Name of the aggregator
        **kwargs: Additional arguments for the aggregator
        
    Returns:
        Aggregator instance
    """
    normalized = name.lower().replace("-", "").replace("_", "")
    aggregators = {
        "mean": MeanAggregator,
        "upgrad": UPGradAggregator,
        "mgda": MGDAAggregator,
        "cagrad": CAGradAggregator,
        "pcgrad": PCGradAggregator,
        "dualproj": lambda **kw: TorchJDAggregator("DualProj", **kw),
        "alignedmtl": lambda **kw: TorchJDAggregator("AlignedMTL", **kw),
        "graddrop": lambda **kw: TorchJDAggregator("GradDrop", **kw),
        "imtlg": lambda **kw: TorchJDAggregator("IMTLG", **kw),
        "nashmtl": lambda **kw: TorchJDAggregator("NashMTL", **kw),
        "rgw": lambda **kw: TorchJDAggregator("Random", **kw),
    }
    alias = {
        "amean": "mean",
        "aupgrad": "upgrad",
        "amgda": "mgda",
        "acagrad": "cagrad",
        "apcgrad": "pcgrad",
        "adualproj": "dualproj",
        "aalignedmtl": "alignedmtl",
        "agraddrop": "graddrop",
        "aimtlg": "imtlg",
        "anashmtl": "nashmtl",
        "argw": "rgw",
    }
    key = alias.get(normalized, normalized)

    if key not in aggregators:
        raise ValueError(
            f"Unknown aggregator: {name}. Available: {list(aggregators.keys())} "
            f"(accepted aliases: {list(alias.keys())})"
        )

    return aggregators[key](**kwargs)


def get_torchjd_aggregator(name: str, **kwargs):
    """
    Factory for torchjd.aggregation classes (no Jacobian stacking wrapper).
    """
    if tjagg is None:
        raise ImportError("torchjd is required for torchjd backend aggregators.")

    normalized = name.lower().replace("-", "").replace("_", "")
    mapping = {
        "mean": "Mean",
        "sum": "Sum",
        "upgrad": "UPGrad",
        "mgda": "MGDA",
        "cagrad": "CAGrad",
        "pcgrad": "PCGrad",
        "dualproj": "DualProj",
        "alignedmtl": "AlignedMTL",
        "graddrop": "GradDrop",
        "imtlg": "IMTLG",
        "nashmtl": "NashMTL",
        "rgw": "Random",
        "random": "Random",
    }
    class_name = mapping.get(normalized)
    if class_name is None:
        raise ValueError(f"Unsupported torchjd aggregator type: {name}")
    cls = getattr(tjagg, class_name, None)
    if cls is None:
        raise ValueError(f"torchjd.aggregation has no class '{class_name}'.")
    if not kwargs:
        return cls()
    try:
        import inspect

        sig = inspect.signature(cls.__init__)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return cls(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return cls(**filtered)
    except Exception:
        # Fallback: attempt without kwargs
        return cls()
