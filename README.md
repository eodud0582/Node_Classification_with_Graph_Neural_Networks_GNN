# Node Classification with Graph Neural Networks

## Competition Results

<img width=25% alt="image" src="https://github.com/user-attachments/assets/d4f26c1f-118a-4a87-9034-33aa8aeb5a9b" />

Our team took 1st place in the competition.

---
## Objective
Our goal is to develop a highly accurate node classification model for a given graph dataset, predicting each node’s class label based on its features and relationships. We conducted a comparative study of modern graph neural network (GNN) architectures to identify which models best utilize node features and graph structure. The final performance was evaluated on a hidden test set.

## Dataset Overview

The dataset contains 2,480 nodes connected via an adjacency matrix with 10,100 edges across 7 classes. Each node has 1,390 features, with only 496 nodes having known labels (used for training) and 1,984 unlabeled nodes for testing.

| Nodes | Edges | Classes | Features | Train / Test |
| --- | --- | --- | --- | --- |
| 2,480 | 10,100 | 7 | 1,390 | 496 / 1,984 |

## Final Result

With these parameters, our model was able to reach a top internal cross-validated accuracy of 90.32% and a testing accuracy of 85.23%.
