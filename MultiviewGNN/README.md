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

### C. Feature Fusion
* The outputs of all three branches are **concatenated** (stacked together).
* A final Multi-Layer Perceptron (MLP) processes this combined vector to classify the node.

---

## 2. Training Strategy & Pipeline

Building a strong architecture is only half the battle. The training pipeline includes several advanced techniques to handle data scarcity and class imbalance.

### A. Robustness & Augmentation
* **Noise Injection:** During training, we add small Gaussian noise (`std=0.02`) to the node features. This acts as data augmentation, forcing the model to learn robust features rather than memorizing the training data.
* **DropEdge:** We randomly remove 10% of edges during training (`p=0.1`). This prevents the model from over-relying on specific connections.

### B. Handling Class Imbalance
The dataset is heavily imbalanced. To fix this:
* **Class Weights:** The Loss Function (`NLLLoss`) is weighted. Errors on rare classes are penalized more heavily.
* **Oversampling:** Inside each training fold, we randomly duplicate samples from minority classes so that the model sees an equal distribution of labels.

### C. Stratified K-Fold Cross-Validation
* We use **5-Fold Stratified CV**. This ensures that every fold maintains the same percentage of samples for each class as the complete set, providing a reliable performance metric.

---

## 3. Advanced Semi-Supervised Learning

Since only about 20% of the data is labeled, we implemented a **Pseudo-Labeling (Self-Training)** loop to utilize the unlabeled test data.

### Step 1: Teacher Training
We train the `MultiviewGNN` on the known training data and predict labels for the test set.

### Step 2: High-Confidence Selection
We look at the model's confidence (probability) for each prediction.
* **Threshold:** `0.98` (98% confidence).
* If the model is 98% sure about a test node, we assume that label is correct and add it to the training set.

### Step 3: Student Retraining & Ensemble
* We retrain the model from scratch using the **expanded dataset** (Original Train + Pseudo-Labeled Test).
* **Final Ensemble:** To reduce variance, the final submission is a **Majority Vote** between the Cross-Validation predictions and the Pseudo-Labeled model's predictions.

---

## Files in this Directory

* `multiview_gnn.py`: The complete source code, including the model class, training loops, and pseudo-labeling logic.
