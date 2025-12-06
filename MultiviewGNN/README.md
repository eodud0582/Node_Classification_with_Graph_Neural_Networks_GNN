# MultiviewGNN: A Hybrid Graph Neural Network

This directory contains the implementation of **MultiviewGNN**, a custom architecture designed to maximize node classification performance by combining multiple graph convolution paradigms.

Instead of relying on a single type of GNN (like GCN or GAT), this model adopts a **"multi-view" approach**. It processes the graph through parallel branches—each designed to capture different structural and feature-based signals—and fuses them for the final prediction.

---
## 1. Model Architecture

The core philosophy is that no single GNN layer is perfect for all nodes. Some nodes require attention mechanisms (GAT), while others benefit from neighborhood aggregation (SAGE) or deep structural propagation (ResGCN).

<p align="center"> <img width="50%"  height="1151" alt="image" src="https://github.com/user-attachments/assets/d9bc349c-cba9-4dee-833b-30dd883ed22c" />

### A. Input Processing & Structural Encoding
Before entering the main layers, the data undergoes specific preprocessing:
* **Robust Scaling:** We use `RobustScaler` to handle outliers in the node features.
* **Degree Embedding:** The model explicitly learns from the graph structure by embedding the **node degree** (number of connections).
    * *Why?* A hub node with 100 connections behaves differently than an isolated node. Standard GNNs sometimes lose this context, so we inject it explicitly.
    * *Logic:* `Input Features + (Degree Embedding * 0.1)`

### B. The Multi-View Parallel Branches
The input is fed simultaneously into three distinct branches. The outputs are concatenated at the end.

1.  **GATv2 Branch (Attention View)**
    * **Layer:** `GATv2Conv` with Multi-head Attention.
    * **Role:** Learns *which* neighbors are most important. It helps the model focus on relevant connections and ignore noisy edges.
2.  **ResGCN Branch (Structural View)**
    * **Layer:** Custom `ResGCNBlock`.
    * **Mechanism:** Standard GCN layers combined with **Residual Connections** (skip-connections) and **Graph Normalization**.
    * **Role:** Allows the model to go deeper without suffering from the vanishing gradient problem. It captures the global structure effectively.
3.  **GraphSAGE Branch (Spatial View)**
    * **Layer:** `SAGEConv`.
    * **Role:** Aggregates neighborhood information efficiently. GraphSAGE is excellent at generalizing to nodes that have slightly different local structures.

**Why this combination? (The Synergy)**

While GAT focuses on specific important neighbors, GraphSAGE generalizes well over neighborhood averages, and ResGCN preserves global structural signals through deeper layers.

By concatenating these views, the model becomes robust: if the attention mechanism fails on a noisy edge, the ResGCN branch still provides a reliable baseline signal. This prevents the "over-smoothing" problem often seen in single-architecture GNNs.

### C. Feature Fusion
* The outputs of all three branches are **concatenated** (stacked together).
* A final Multi-Layer Perceptron (MLP) processes this combined vector to classify the node.

---

## 2. Key Technical Decisions

Beyond standard layers, specific architectural choices were made to address common GNN limitations:

### A. Mitigating Oversmoothing (Residual Connections)

In deep GNNs, node features tend to become indistinguishable after multiple aggregation steps (the "oversmoothing" problem).
* Solution: We implemented Residual Connections (Skip-Connections) in the ResGCN branch.
* Effect: By adding the input features directly to the output ($x_{out} = F(x) + x$), the model preserves unique node identities even as it captures global structural information.

### B. Stabilizing Training (Graph Normalization)

Standard Batch Normalization assumes independent and identically distributed (i.i.d) samples, which is not always true for graph nodes.
* Solution: We used GraphNorm instead of BatchNorm.
* Effect: GraphNorm normalizes features across the nodes within a specific graph/subgraph. This is empirically shown to converge faster and generalize better for node classification tasks compared to standard normalization techniques.

### C. Dynamic Attention (GATv2 over GAT)

* Why GATv2? Standard GAT computes static attention (the ranking of neighbors depends only on the global weight matrix, not the query node).
* Effect: We utilized GATv2, which introduces dynamic attention where the importance of a neighbor is conditioned on both the source and target nodes. This makes the attention mechanism strictly more expressive and capable of handling complex edge cases.

---

## 3. Training Strategy & Pipeline

Building a strong architecture is only half the battle. The training pipeline includes several advanced techniques to handle data scarcity and class imbalance.

### A. Robustness & Augmentation
* **Noise Injection:** During training, we add small Gaussian noise (`std=0.02`) to the node features. This acts as data augmentation, forcing the model to learn robust features rather than memorizing the training data.
* **DropEdge:** We randomly remove 10% of edges during training (`p=0.1`). This prevents the model from over-relying on specific connections.

### B. Handling Class Imbalance
The dataset is heavily imbalanced. We address this carefully to avoid data leakage:
* **Class Weights:** The Loss Function (`NLLLoss`) is weighted (assigned class weights inversely proportional to class frequencies). Errors on rare classes are penalized more heavily.
* **Oversampling:** Inside each **training fold** (training fold only), we randomly duplicate samples from minority classes so that the model sees an equal distribution of labels.
   - We do **not** oversample the dataset before splitting. This ensures that synthetic copies of validation nodes never leak into the training set, guaranteeing that our Cross-Validation score remains honest and realistic.

### C. Stratified K-Fold Cross-Validation
* We use **5-Fold Stratified CV**. This ensures that every fold maintains the same percentage of samples for each class as the complete set, providing a reliable performance metric.

### D. Reproducibility & Determinism

GNNs are often sensitive to weight initialization and GPU non-determinism. To guarantee that our results are reproducible:
* Full Seed Fixing: We explicitly set random seeds (777) for Python, NumPy, PyTorch, and CUDA.
* Deterministic Algorithms: We forced CuDNN to use deterministic algorithms (torch.use_deterministic_algorithms(True)), ensuring that the model converges to the exact same result across different runs on the same hardware.

---

## 4. Advanced Semi-Supervised Learning

Since only about 20% of the data is labeled, We implemented a **Pseudo-Labeling (Self-Training)** loop to utilize the unlabeled test data.

### Step 1: Teacher Training
We train the `MultiviewGNN` on the known training data and predict labels for the test set.

### Step 2: High-Confidence Selection
We look at the model's confidence (probability) for each prediction.
* **Threshold:** `0.98` (98% confidence).
* If the model is 98% sure about a test node, We assume that label is correct and add it to the training set.

### Step 3: Student Retraining & Ensemble
* We retrain the model from scratch using the **expanded dataset** (Original Train + Pseudo-Labeled Test).
* **Fresh Initialization:** We re-initialize the model weights rather than fine-tuning to avoid getting stuck in local minima from the previous stage.
* **Pure Validation Set:** Crucially, the validation set used during this phase contains only original ground-truth labels. We intentionally exclude pseudo-labels from validation to ensure we are optimizing for real accuracy, not just fitting our own guesses.
* **Final Ensemble:** To reduce variance, the final submission is a **Majority Vote** between the Cross-Validation predictions and the Pseudo-Labeled model's predictions.

---

## 5. Model Configuration

The following hyperparameters were selected based on extensive tuning:

| Hyperparameter        | Value | Note |
|-----------------------|-------|------|
| Hidden Dimension      | 256   | High capacity to capture complex patterns |
| Dropout               | 0.3   | Lower than default (0.5) to preserve structural info |
| Learning Rate         | 0.001 | With ReduceLROnPlateau Scheduler |
| Weight Decay          | 5e-4  | L2 Regularization to prevent( overfitting |
| Heads (GATv2)         | 4     | Multi-head attention for richer feature learning |
| Degree Clip           | 99    | Clamped max degree for embedding stability |
| Pseudo-Label Threshold| 0.98  | Only very high-confidence samples are used |

---

## Files in this Directory

* `multiview_gnn.py`: The complete source code, including the model class, training loops, and pseudo-labeling logic.
