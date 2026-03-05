# Jacobian Descent CIFAR-10 Reproduction

This repository reproduces the CIFAR-10 experiments from the paper:
**"Jacobian Descent for Multi-Objective Optimization"** (arXiv:2406.16232)

## Paper Information
- **Authors**: Pierre Quinton, Valérian Rey
- **Institution**: EPFL
- **arXiv**: https://arxiv.org/abs/2406.16232
- **Code**: https://github.com/TorchJD/torchjd

## Experiment Setup

### Dataset
- **CIFAR-10**: 50,000 training images, 10,000 test images
- **Multi-task setup**: 5 binary classification tasks
  - Task 0: Classes 0,1 vs others
  - Task 1: Classes 2,3 vs others
  - Task 2: Classes 4,5 vs others
  - Task 3: Classes 6,7 vs others
  - Task 4: Classes 8,9 vs others

### Model Architecture
- **Backbone**: ResNet-18 (modified for CIFAR-10)
- **Task heads**: 5 separate linear layers (one per task)
- **Input size**: 32×32 RGB images

### Training Configuration
- **Epochs**: 200
- **Batch size**: 128
- **Learning rate**: 0.1 (with cosine annealing)
- **Optimizer**: SGD with momentum 0.9
- **Weight decay**: 5e-4
- **Data augmentation**: Random crop, horizontal flip

### Aggregators Implemented
1. **Mean**: Simple average of task gradients (baseline)
2. **UPGrad**: Unconflicting Projection Gradient (paper's main method)
3. **MGDA**: Multiple Gradient Descent Algorithm
4. **CAGrad**: Conflict-Averse Gradient
5. **PCGrad**: Project Conflicting Gradients

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run full experiment
```bash
python run_experiment.py --config config.yaml
```

### Run with specific device
```bash
python run_experiment.py --device cuda
```

### Quick test (fewer epochs)
Edit `config.yaml` to change `num_epochs` to a smaller value (e.g., 10).

## Output

Results are saved in the `results/` directory:
- `results.json`: Numerical results
- `results.png`: Training curves plot
- `checkpoints/`: Model checkpoints

## Expected Results

Based on the paper, UPGrad should achieve:
- **Average accuracy**: ~85-90% across all tasks
- **Fairness**: Low variance between task accuracies
- **Convergence**: Faster than baseline methods

## File Structure

```
JacobianDescentMOOExperiment/
├── aggregators.py          # Multi-objective gradient aggregators
├── run_experiment.py       # Main experiment script
├── config.yaml            # Experiment configuration
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── results/              # Output directory (created during run)
    ├── results.json
    ├── results.png
    └── checkpoints/
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{quinton2024jacobian,
  title={Jacobian Descent for Multi-Objective Optimization},
  author={Quinton, Pierre and Rey, Val{\'e}rian},
  journal={arXiv preprint arXiv:2406.16232},
  year={2024}
}
```

## Notes

- This is a reproduction effort and may not exactly match the paper's results
- Some implementation details may differ from the original
- For the official implementation, see https://github.com/TorchJD/torchjd
