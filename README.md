# Node Classification with Graph Neural Networks (GNN)

## Competition Results

<img width=25% alt="image" src="https://github.com/user-attachments/assets/d4f26c1f-118a-4a87-9034-33aa8aeb5a9b" />

Our team took 1st place in the course competition.

---

## Project Overview
This project aims to build a high-accuracy node classification model using Graph Neural Networks (GNNs). Graph data structures are used to model connections, such as users on social media or fraud detection in financial systems. Our goal was to predict the class label of nodes based on their features and relationships within the graph.

## Dataset Summary
We worked with a graph dataset containing nodes (entities) and edges (interactions). A major challenge was the limited number of labeled nodes for training.

| Metric | Nodes | Edges | Classes | Features | Train/Test Split |
| --- | --- | --- | --- | --- | --- |
| Value | 2,480 | 10,100 | 7 | 1,390 | 496 / 1,984 (Only about 20% of data was labeled) |

## Methodology and Exploration
Our team adopted a comprehensive approach, testing various preprocessing techniques, models, and training strategies.

### 1. Data Preprocessing
To improve data quality, we experimented with:

- Feature Selection: Removed features with zero standard deviation.
- Normalization: Used StandardScaler and RobustScaler (to handle outliers).
- Half-Hop and Two-Hop: Aggregated information to capture local and wider context.
- Virtual Node: Added a central node to connect distant parts of the graph.
- RandomWalk (NodeWalk): Captured community structures through random path sampling.
- DropEdge: Randomly removed edges during training to prevent overfitting.

### 2. Model Exploration
We compared several GNN architectures using PyTorch:

- GCN / GCNII: Baseline models using graph convolution.
- GraphSAGE: Efficient sampling of neighbors for faster training.
- GAT / GATv2: Uses attention mechanisms to prioritize important neighbors.
- MultiviewGNN: An ensemble approach combining GAT, GCN, and GraphSAGE.
- APPNP (Final Choice): Uses PageRank to propagate information, balancing local and global features effectively.

### 3. Training Enhancements
To handle class imbalance and small data size:

- Oversampling and GraphSMOTE: Generated synthetic data for rare classes.
- Pseudo-labeling: Used high-confidence predictions on unlabeled data to expand the training set.
- Hyperparameter Tuning: Used Optuna for automated optimization and manual Grid Search.

## Final Implementation
After extensive testing, the APPNP (Approximate Personalized Propagation of Neural Predictions) model achieved the best performance, with MultiviewGNN with pseudo-labeling coming in very close.

### Architecture Details
- Input: Z-score scaled features.
- Layers: Two Linear layers with ReLU activation and Dropout.
- Propagation: APPNP layer (K=72 hops, alpha approx 0.05).
- Optimizer: Adam with ReduceLROnPlateau scheduler.

### Performance Results
- Cross-Validation Accuracy (Internal): 90.32%
- Test Accuracy (Hidden Set): 85.23%

### Optimal Hyperparameters

| Hidden Dimension | Dropout Rate | Learning Rate | Weight Decay | Propagation Hops (K) | Teleport Prob (Alpha) |
| --- | --- | --- | --- | --- | --- |
| 64 | 0.507 | 7.75e-3 | 9.79e-7 | 72 | 0.049 |

## My Contributions
- Developed MultiviewGNN, implemented Pseudo-labeling and ensemble methods.

## Tech Stack
- Language: Python
- Frameworks: PyTorch, PyTorch Geometric
- Tools: Optuna (Tuning), Scikit-learn (Preprocessing)
