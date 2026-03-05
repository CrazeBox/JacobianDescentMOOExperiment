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


def _solve_problem_quietly(prob):
    """Solve CVXPY problem while suppressing noisy solver stdout/stderr."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            prob.solve(solver=cp.OSQP, verbose=False, polish=False, warm_start=True)
        except Exception:
            prob.solve(verbose=False)


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
        self.epsilon = epsilon
    
    def __call__(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        if len(gradients) == 1:
            return gradients[0]
        
        # Stack gradients into Jacobian matrix J [num_tasks, num_params]
        J = torch.stack(gradients)
        m = J.size(0)  # number of tasks
        
        # Compute Gram matrix G = J @ J^T
        G = torch.matmul(J, J.t())
        
        # Add regularization for numerical stability
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
                # Convert to numpy for CVXPY
                G_np = G.cpu().numpy()
                
                # Define optimization variable
                w = cp.Variable(m)
                
                # Define objective: minimize 0.5 * w^T G w - sum(w)
                objective = cp.Minimize(0.5 * cp.quad_form(w, G_np) - cp.sum(w))
                
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
        except Exception as e:
            print(f"QP solver failed: {e}, using uniform weights")
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
        self.epsilon = epsilon
    
    def __call__(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        if len(gradients) == 1:
            return gradients[0]
        
        J = torch.stack(gradients)
        m = J.size(0)
        
        # Compute Gram matrix
        G = torch.matmul(J, J.t())
        G = G + self.epsilon * torch.eye(m, device=G.device)
        
        try:
            if cp is None:
                # Fallback when cvxpy is unavailable.
                weights = torch.ones(m, dtype=J.dtype, device=J.device) / m
            else:
                # Solve min ||sum(w_i * grad_i)||^2 s.t. sum(w_i) = 1, w_i >= 0
                G_np = G.cpu().numpy()
                
                w = cp.Variable(m)
                
                # Objective: minimize w^T G w (norm of weighted sum)
                objective = cp.Minimize(cp.quad_form(w, G_np))
                
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
        except Exception as e:
            print(f"MGDA solver failed: {e}, using uniform weights")
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
        self.c = c
        self.epsilon = epsilon
    
    def __call__(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        if len(gradients) == 1:
            return gradients[0]
        
        J = torch.stack(gradients)
        m = J.size(0)
        
        # Compute average gradient
        avg_grad = J.mean(dim=0)
        
        # Adjust individual gradients
        adjusted_grads = []
        for grad in gradients:
            # Compute dot product
            dot = torch.dot(grad, avg_grad)
            
            # If conflicting (dot < 0), project away from conflict
            if dot < -self.epsilon:
                # Project grad onto avg_grad
                proj = (dot / (torch.norm(avg_grad)**2 + self.epsilon)) * avg_grad
                adjusted = grad - (1 + self.c) * proj
            else:
                adjusted = grad
            
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
        self.epsilon = epsilon
    
    def __call__(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        if len(gradients) == 1:
            return gradients[0]
        
        # Start with a copy of gradients
        adjusted_grads = [g.clone() for g in gradients]
        
        # Pairwise projection
        for i in range(len(adjusted_grads)):
            for j in range(len(adjusted_grads)):
                if i != j:
                    grad_i = adjusted_grads[i]
                    grad_j = adjusted_grads[j]
                    
                    # Compute dot product
                    dot = torch.dot(grad_i, grad_j)
                    
                    # If conflicting, project
                    if dot < -self.epsilon:
                        # Project grad_i onto grad_j
                        proj = (dot / (torch.norm(grad_j)**2 + self.epsilon)) * grad_j
                        adjusted_grads[i] = grad_i - proj
        
        # Return mean of adjusted gradients
        aggregated = torch.stack(adjusted_grads).mean(dim=0)
        return aggregated


def get_aggregator(name: str, **kwargs) -> Aggregator:
    """
    Factory function to get aggregator by name.
    
    Args:
        name: Name of the aggregator
        **kwargs: Additional arguments for the aggregator
        
    Returns:
        Aggregator instance
    """
    aggregators = {
        "mean": MeanAggregator,
        "upgrad": UPGradAggregator,
        "mgda": MGDAAggregator,
        "cagrad": CAGradAggregator,
        "pcgrad": PCGradAggregator
    }
    
    if name.lower() not in aggregators:
        raise ValueError(f"Unknown aggregator: {name}. Available: {list(aggregators.keys())}")
    
    return aggregators[name.lower()](**kwargs)
