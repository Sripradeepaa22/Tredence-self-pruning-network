# 🧠 Self-Pruning Neural Network (PyTorch)

A PyTorch implementation of a neural network that **learns to prune itself** during training using learnable gate scores to automatically induce sparsity — no post-training pruning required.

Trained and evaluated on **CIFAR-10**, this project demonstrates how to balance classification accuracy with model efficiency by tuning a sparsity penalty (λ).

---

## 📌 Overview

Unlike traditional pruning (which typically happens *after* training), this implementation features a `PrunableLinear` layer that learns which weights are important **during** the training phase itself. A sigmoid-based gating mechanism combined with a gradual sparsity scheduler causes redundant connections to "shut off" progressively — letting the model first learn useful features before losing parameters.

---

## ✨ Key Features

- **Dynamic Gating** — Every connection in the linear layers has an associated learnable gate score.
- **Gradual Pruning** — A λ scheduler ramps up sparsity pressure over training epochs, preventing premature pruning.
- **High Sparsity** — Achieves **>95% sparsity** on CIFAR-10 while maintaining competitive MLP-level accuracy.

---

## 🏗️ Architecture

The model is a Multi-Layer Perceptron (MLP) with three custom `PrunableLinear` layers:

| Layer | Input → Output | Description |
|-------|---------------|-------------|
| Input | 3072 → 512 | Flattened 32×32×3 CIFAR-10 images |
| Hidden | 512 → 256 | Intermediate representation |
| Output | 256 → 10 | One logit per CIFAR-10 class |

## The Pruning Mechanism
The core of this project is the PrunableLinear class. Its forward pass is defined as:
Output=x⋅(W⊙σ(gate_scores×1.5))+b

Where:
- ⊙ is element-wise (Hadamard) product σ\sigma
- σ is the sigmoid activation function
- gate_scores are learnable parameters that control which weights stay active

As training progresses, gate scores polarize toward 0 (pruned) or 1 (active), creating a sparse but effective network.

## 📊 Results

Experiments with varying λ values reveal the accuracy–sparsity trade-off:

<img width="393" height="83" alt="image" src="https://github.com/user-attachments/assets/17a8e2fc-88eb-4acd-8ca3-2abbbd38cbe0" />


*

> Higher λ = more aggressive pruning, lower accuracy. Tune to your use case.

---

## 📦 Prerequisites & Installation

**Requires Python 3.12+**

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

---

## 🚀 Usage

The training script includes a `run_experiment` function that handles the full training loop, gradual λ scaling, and evaluation.

```python
from self_pruning_network import run_experiment

# lambda_val controls the strength of the sparsity penalty
result = run_experiment(lambda_val=0.005, epochs=20)

print(f"Final Accuracy: {result['accuracy']}%")
print(f"Sparsity:       {result['sparsity']}%")
```

---

## 📈 Visualizations

The notebook generates two diagnostic plots:

### Gate Value Distribution
Shows how gate scores polarize toward 0 (pruned) or 1 (active) over training — a healthy sign of effective pruning.

<img width="597" height="455" alt="image" src="https://github.com/user-attachments/assets/81cc8160-e0bf-4764-bc7b-cdf63c915d50" />

### λ vs. Performance
Illustrates the trade-off between sparsity pressure and final classification accuracy across multiple experiments.
<img width="562" height="459" alt="image" src="https://github.com/user-attachments/assets/522b3aba-52b9-4791-815c-1c408911f228" />

---

## 🛠️ How the λ Scheduler Works

Rather than applying full sparsity pressure from epoch 1, the scheduler gradually increases λ from `0` to its target value over training:

```
Effective λ at epoch t = λ_target × (t / total_epochs)
```

This ensures the model first develops useful representations before being forced to compress them — critical for avoiding premature convergence to a useless sparse solution.

---

## 📁 Project Structure

```
self-pruning-network/
├── notebook.ipynb            # Full walkthrough with plots
└── README.md
```

---

## 📄 License

MIT License. See `LICENSE` for details.



