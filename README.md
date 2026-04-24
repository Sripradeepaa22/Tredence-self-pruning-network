Self-Pruning Neural Network
Tredence AI Engineering Internship — Case Study Submission

📌 Problem Overview
Design and implement a neural network that learns to prune itself during training — not as a post-training step, but dynamically, using learnable gate parameters attached to every weight.
The core idea:

Every weight w_ij gets a learnable scalar gate g_ij ∈ (0, 1)
If a gate → 0, that weight is effectively removed from the network
Training jointly optimises classification accuracy + sparsity via a custom loss

Dataset: CIFAR-10 (60,000 images, 10 classes, 32×32×3)

🏗️ Architecture
Input (32 × 32 × 3 = 3072)
        │
        ▼
 PrunableLinear(3072 → 512)   ← gate_scores learned here
        │  ReLU
        ▼
 PrunableLinear(512 → 256)    ← gate_scores learned here
        │  ReLU
        ▼
 PrunableLinear(256 → 10)     ← gate_scores learned here
        │
        ▼
   Class Logits → CrossEntropyLoss
The PrunableLinear Layer
Each layer holds three learnable parameter tensors:
ParameterShapeRoleweight(out_features, in_features)Standard linear weightsbias(out_features,)Standard biasgate_scores(out_features, in_features)Learnable pruning gates
Forward pass:
pythongates          = torch.sigmoid(gate_scores * 1.5)   # ∈ (0, 1), sharpened
pruned_weights = weight * gates                      # element-wise mask
output         = F.linear(x, pruned_weights, bias)
Gradients flow automatically through both weight and gate_scores via PyTorch autograd — no custom backward needed.

⚙️ Training Objective
Total Loss = CrossEntropyLoss(logits, labels) + λ × SparsityLoss

SparsityLoss = mean(sigmoid(gate_scores))   # L1-style penalty on all gates
Why L1 Encourages Sparsity
PenaltyGradient near zeroEffectL2 (sum of squares)Decays → 0Shrinks weights but rarely zeros themL1 (sum of values)ConstantPushes values to exactly 0
Because sigmoid(x) ∈ (0,1) is always positive, the L1 norm is just the sum of gate values. The optimiser receives a constant gradient push toward zero regardless of how small a gate already is — this is why gates collapse to exactly 0 rather than staying "small but non-zero."
Gradual Lambda Warmup
pythoncurrent_lambda = lambda_val × (epoch / total_epochs)
Lambda is annealed from 0 → λ over training. This lets the model first learn to classify, then progressively learn to prune — preventing the sparsity loss from overwhelming classification before useful features are learned.

📊 Results
Lambda (λ)Test AccuracySparsity Level0.00552.37%95.56%0.0151.82%96.18%
Key Observations

Higher λ → higher sparsity, slightly lower accuracy — the expected trade-off
Both models achieve >95% sparsity, meaning over 95% of all weight connections are pruned to near-zero
The accuracy difference between λ=0.005 and λ=0.01 is only ~0.55%, while sparsity increases by ~0.62% — showing the model is very robust to aggressive pruning
Gradual warmup prevents accuracy collapse that occurs with fixed high-λ training

Training Progression (λ = 0.005)
EpochLossSparsity1801.9293.20%5635.6493.36%10566.2293.92%15524.4894.72%20492.6095.56%
Sparsity grows steadily and consistently as training progresses — exactly the expected self-pruning behaviour.

📈 Plots
The notebook generates two plots:
1. Gate Value Distribution (Best Model)

Shows a large spike near 0 — the majority of gates are pruned
A smaller cluster near 0.5–1.0 — the surviving important connections
This bimodal distribution is the signature of successful self-pruning

2. Lambda vs Accuracy & Sparsity

Shows the trade-off curve as λ increases
Demonstrates that the sparsity mechanism works controllably


🚀 How to Run
Google Colab (Recommended)

Go to colab.research.google.com
Upload self_pruning_network.ipynb
Runtime → Change runtime type → T4 GPU ✅
Runtime → Run all


GPU gives ~5× speedup. Full run takes ~10–15 minutes on T4.

Local
bashgit clone https://github.com/YOUR_USERNAME/tredence-self-pruning-network
cd tredence-self-pruning-network
pip install torch torchvision matplotlib numpy
jupyter notebook self_pruning_network.ipynb
Requirements
Python      3.8+
PyTorch     2.0+
torchvision ≥ 0.15
matplotlib  ≥ 3.5
numpy       ≥ 1.21

Results
<img width="597" height="455" alt="image" src="https://github.com/user-attachments/assets/81cc8160-e0bf-4764-bc7b-cdf63c915d50" />
<img width="562" height="459" alt="image" src="https://github.com/user-attachments/assets/522b3aba-52b9-4791-815c-1c408911f228" />



📁 Repository Structure
tredence-self-pruning-network/
├── self_pruning_network.ipynb   ← Main notebook with full experiment
└── README.md                    ← This file

🛠️ Tech Stack
ToolPurposePython 3.xCore languagePyTorch 2.xModel, autograd, training looptorchvisionCIFAR-10 dataset loadingmatplotlibPlots and visualisationsnumpyNumerical utilities

💡 Design Decisions & Reasoning
DecisionWhysigmoid for gatesSmooth, differentiable, bounded ∈ (0,1) — perfect soft gateNegative init for gate_scores (-1.5)Starts gates near ~0.18 — model opens gates it needs rather than closing ones it doesn'tTemperature scaling (× 1.5)Sharpens sigmoid curve, pushes gates toward more decisive 0 or 1 valuesmean not sum for SparsityLossScale-invariant across layers of different sizesGradual lambda warmupPrevents pruning from collapsing classification before features are learned
