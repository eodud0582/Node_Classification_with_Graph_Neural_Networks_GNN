import os
import random
import json
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, GraphNorm, MessagePassing
from torch_geometric.utils import dropout_edge, add_self_loops, degree, from_scipy_sparse_matrix
from torch_geometric.data import Data
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

# =============================================================================
# 1. Configuration & Reproducibility
# =============================================================================

# Dropout Rate: Controls the regularization strength.
# Based on hyperparameter tuning (optimum was 0.4), but set to 0.3 
# to slightly reduce randomness and preserve more information during training.
DROPOUT = 0.3

def set_seeds(seed=777):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Enforce deterministic algorithms in CUDA (may impact performance but ensures reproducibility)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure CuDNN uses deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

SEED = 777
set_seeds(SEED)
print(f'Random seed set to: {SEED}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 2. Model Components
# =============================================================================

class ResGCNBlock(nn.Module):
    """
    GCN Block with Residual Connection and Graph Normalization.
    Residual connections help prevent vanishing gradients in deeper GNNs.
    """
    def __init__(self, in_channels, out_channels, dropout):
        super(ResGCNBlock, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.norm = GraphNorm(out_channels) # Normalizes node features across the graph
        self.dropout = dropout
        
        # Projection layer to match dimensions for residual addition if needed
        if in_channels != out_channels:
            self.res_conv = nn.Linear(in_channels, out_channels)
        else:
            self.res_conv = nn.Identity()
            
    def forward(self, x, edge_index, batch=None):
        identity = self.res_conv(x) # Save input for residual connection
        
        x = self.gcn(x, edge_index)
        x = self.norm(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x + identity # Add residual (ResNet style)

class EdgeFeatureConv(MessagePassing):
    """
    Custom MessagePassing layer to handle edge features.
    (Note: This class is defined but currently not actively used in the main EnhancedGNN logic,
    preserved for potential future extensions using edge attributes).
    """
    def __init__(self, in_channels, out_channels):
        super(EdgeFeatureConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels * 2, out_channels)
        
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
        
    def message(self, x_i, x_j):
        # Concatenate features of source (j) and target (i) nodes
        edge_features = torch.cat([x_i, x_j], dim=1)
        return self.lin(edge_features)

class EnhancedGNN(nn.Module):
    """
    The Core Backbone Model: Multi-view GNN Architecture.
    Combines GAT (Attention), GCN (Spectral), and GraphSAGE (Spatial) 
    to capture different aspects of the graph structure.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3, heads=4, layers=3):
        super(EnhancedGNN, self).__init__()
        self.dropout = dropout
        self.layers = layers
        self.hidden_channels = hidden_channels
        
        # 1. Input Transformation: Project features to hidden dimension
        self.input_transform = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Parallel Branches
        self.gat_convs = nn.ModuleList()
        self.gcn_blocks = nn.ModuleList()
        self.sage_convs = nn.ModuleList()
        
        # Initialize the first layer of each branch
        self.gat_convs.append(GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        self.gcn_blocks.append(ResGCNBlock(hidden_channels, hidden_channels, dropout))
        self.sage_convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Initialize subsequent layers
        for _ in range(1, layers):
            self.gat_convs.append(GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
            self.gcn_blocks.append(ResGCNBlock(hidden_channels, hidden_channels, dropout))
            self.sage_convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # 3. Output Layer: Aggregates features from all branches
        # Input dim is *3 because we concatenate GAT + GCN + SAGE outputs
        self.output = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        # 4. Degree Embedding: Learnable embedding based on node degree (structure bias)
        self.degree_emb = nn.Embedding(100, hidden_channels)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Edge Dropout (DropEdge): Regularization technique during training
        # Randomly removes edges to prevent overfitting to specific graph structures
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=0.1)
        
        # Add Self-loops & Compute Degree
        edge_index_self, _ = add_self_loops(edge_index)
        deg = degree(edge_index_self[0], x.size(0)).long().clamp(max=99)
        deg_emb = self.degree_emb(deg)
        
        # Initial Transform + Add Degree Embedding
        x = self.input_transform(x)
        x = x + deg_emb * 0.1 # Weighting the structural info slightly
        
        # Split into parallel branches
        gat_x, gcn_x, sage_x = x, x, x
        
        for i in range(self.layers):
            # Branch 1: GATv2
            gat_x = self.gat_convs[i](gat_x, edge_index)
            gat_x = F.relu(gat_x)
            gat_x = F.dropout(gat_x, p=self.dropout, training=self.training)
            
            # Branch 2: ResGCN (Block handles activation internally)
            gcn_x = self.gcn_blocks[i](gcn_x, edge_index)
            
            # Branch 3: GraphSAGE
            sage_x = self.sage_convs[i](sage_x, edge_index)
            sage_x = F.relu(sage_x)
            sage_x = F.dropout(sage_x, p=self.dropout, training=self.training)
        
        # Concatenate representations from all views
        x = torch.cat([gat_x, gcn_x, sage_x], dim=1)
        
        # Final prediction
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

class MultiviewGNN(nn.Module):
    """
    Wrapper class for the EnhancedGNN. 
    Keeps the interface clean for training loops.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(MultiviewGNN, self).__init__()
        self.model = EnhancedGNN(
            in_channels, hidden_channels, out_channels, 
            dropout=dropout, heads=4, layers=3
        )
        
    def forward(self, data):
        return self.model(data)


# =============================================================================
# 3. Training & Evaluation Functions
# =============================================================================

def train_model(model, optimizer, scheduler, train_idx, val_idx, data, class_weights=None, 
                epochs=300, patience=50, verbose=True):
    """
    Standard PyTorch training loop with:
    - Weighted Loss (for imbalance)
    - Gradient Clipping (stability)
    - Learning Rate Scheduler
    - Early Stopping (prevents overfitting)
    """
    best_val_acc = 0
    best_test_pred = None
    trigger_times = 0 # Counter for early stopping
    
    # NLLLoss is used because the model outputs log_softmax
    criterion = nn.NLLLoss(weight=class_weights) if class_weights is not None else nn.NLLLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(data)
        loss = criterion(out[train_idx], data.y[train_idx])
        
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_acc = evaluate(model, data, train_idx)
        val_acc = evaluate(model, data, val_idx)
        
        # Adjust Learning Rate based on validation accuracy
        scheduler.step(val_acc)
        
        if verbose and epoch % 20 == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}')
        
        # Early Stopping Logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
            
            # Save predictions for the test set at the best validation epoch
            model.eval()
            with torch.no_grad():
                test_idx = getattr(data, 'test_idx', None)
                if test_idx is not None:
                    best_test_pred = model(data).argmax(dim=1).cpu().numpy()[test_idx.cpu().numpy()]
        else:
            trigger_times += 1
            if trigger_times >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break
    
    return best_val_acc, best_test_pred

def evaluate(model, data, indices):
    """Calculates accuracy on a specific set of indices (Train/Val/Test)."""
    if len(indices) == 0:
        return 0.0
    
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        correct = (pred[indices] == data.y[indices]).sum().item()
        
    return correct / len(indices)


# =============================================================================
# 4. Semi-Supervised Techniques (Pseudo-Labeling)
# =============================================================================

def get_pseudo_labels(model, data, idx_test, confidence_threshold=0.98):
    """
    Generates pseudo-labels for test data.
    Only predictions with probability > confidence_threshold are selected.
    This expands the training set with high-confidence unlabelled data.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = torch.exp(logits) # Convert log_softmax to probability
        max_probs, pseudo_labels = probs.max(dim=1)
        
        # Filter by confidence threshold
        confident_mask = max_probs > confidence_threshold
        confident_idx = idx_test[confident_mask[idx_test].cpu().numpy()]
        confident_labels = pseudo_labels[confident_idx]
        
    return confident_idx, confident_labels

def evaluate_pseudo_model_with_kfold(data, idx_train_orig, confident_idx, confident_labels, 
                                     input_dim, hidden_dim, output_dim, class_weights, 
                                     n_splits=5, seed=777):
    """
    Runs K-Fold Cross Validation specifically including the Pseudo-Labeled data.
    Ensures that the model generated via Pseudo-labeling is robust.
    """
    print("\nStarting Cross-Validation with Pseudo-Labels...")
    
    # Combine original training data with new pseudo-labeled data
    combined_idx = np.concatenate([idx_train_orig, confident_idx])
    combined_labels = np.concatenate([data.y[idx_train_orig].cpu().numpy(), confident_labels.cpu().numpy()])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    fold_val_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(combined_idx, combined_labels)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        
        train_idx = combined_idx[train_idx]
        val_idx = combined_idx[val_idx]
        
        # Vital: Remove pseudo-labeled instances from the Validation Set
        # Validation should only be done on ground-truth data if possible, 
        # or at least kept distinct to avoid leakage.
        val_idx = val_idx[~np.isin(val_idx, confident_idx)]
        if len(val_idx) == 0:
            print(f"Fold {fold+1}: Validation set empty after filtering, skipping.")
            continue
        
        # --- Oversampling Strategy for Imbalance ---
        train_labels = data.y[train_idx].cpu().numpy()
        class_count = np.bincount(train_labels[train_labels >= 0])
        max_size = max(class_count)
        
        oversampled_idx = []
        for cls in range(len(class_count)):
            cls_idx = train_idx[train_labels == cls]
            if len(cls_idx) == 0: continue
            
            # Use fold-specific seed for reproducibility
            np.random.seed(seed + fold) 
            oversampled = np.random.choice(cls_idx, size=max_size, replace=True)
            oversampled_idx.extend(oversampled)
        
        oversampled_idx = np.array(oversampled_idx)
        
        # Initialize fresh model for this fold
        model = MultiviewGNN(
            in_channels=input_dim, hidden_channels=hidden_dim, 
            out_channels=output_dim, dropout=DROPOUT
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=20, factor=0.5, verbose=False
        )
        
        val_acc, _ = train_model(
            model=model, optimizer=optimizer, scheduler=scheduler,
            train_idx=torch.tensor(oversampled_idx).to(device),
            val_idx=torch.tensor(val_idx).to(device),
            data=data, class_weights=class_weights,
            epochs=300, patience=50, verbose=True
        )
        
        fold_val_accs.append(val_acc)
        print(f"Fold {fold+1} Validation Accuracy: {val_acc:.4f}")
    
    mean_acc = np.mean(fold_val_accs)
    std_acc = np.std(fold_val_accs)
    print(f"\nMean Accuracy with Pseudo-Labels: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return mean_acc, std_acc


# =============================================================================
# 5. Main Execution
# =============================================================================

def main():
    print("Loading Data...")
    try:
        base_path = '/content/drive/MyDrive/Colab Notebooks'
        adj = sp.load_npz(f'{base_path}/data/adj.npz')
        feat = np.load(f'{base_path}/data/features.npy')
        labels_subset = np.load(f'{base_path}/data/labels.npy')
        splits = json.load(open(f'{base_path}/data/splits.json'))
    except:
        # Fallback for local environment
        base_path = '.'
        adj = sp.load_npz(f'{base_path}/data/adj.npz')
        feat = np.load(f'{base_path}/data/features.npy')
        labels_subset = np.load(f'{base_path}/data/labels.npy')
        splits = json.load(open(f'{base_path}/data/splits.json'))
    
    idx_train_orig = np.array(splits['idx_train'])
    idx_test = np.array(splits['idx_test'])

    print("Preprocessing Features...")
    # Remove features with zero variance (no information)
    stds = feat.std(axis=0)
    valid_indices = np.where(stds != 0)[0]
    feat = feat[:, valid_indices]
    
    # Dimensionality reduction (PCA)
    # pca = PCA(n_components=0.95, random_state=SEED)
    # feat = pca.fit_transform(feat)
    
    # RobustScaler is better for data with outliers than StandardScaler
    scaler = RobustScaler()
    feat = scaler.fit_transform(feat)
    
    # Prepare Labels (-1 for unlabeled/test nodes)
    full_labels = np.full((feat.shape[0],), -1)
    full_labels[idx_train_orig] = labels_subset
    
    # Convert to PyTorch tensors
    edge_index, _ = from_scipy_sparse_matrix(adj)
    x = torch.FloatTensor(feat)
    y = torch.LongTensor(full_labels)
    
    # --- Data Augmentation: Noise Injection ---
    # Adding small Gaussian noise to features helps generalization
    x_aug = x.clone()
    x_aug += torch.randn_like(x_aug) * 0.02
    
    # Create PyTorch Geometric Data object
    data = Data(x=x_aug, edge_index=edge_index, y=y)
    data = data.to(device)
    data.test_idx = torch.tensor(idx_test, dtype=torch.long).to(device)
    
    # Compute Class Weights to handle imbalance in Loss Function
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_subset), y=labels_subset)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    input_dim = feat.shape[1]
    hidden_dim = 256
    output_dim = len(np.unique(labels_subset))
    
    # ---------------------------------------------------------
    # Phase 1: Initial Cross-Validation with Oversampling
    # ---------------------------------------------------------
    print("\nStarting K-Fold CV on Original Training Data...")
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    fold_val_accs = []
    fold_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(idx_train_orig, labels_subset)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        
        train_idx = idx_train_orig[train_idx]
        val_idx = idx_train_orig[val_idx]
        
        # Dynamic Oversampling for current fold
        train_labels = y[train_idx].cpu().numpy()
        class_count = np.bincount(train_labels)
        max_size = max(class_count)
        
        oversampled_idx = []
        for cls in range(len(class_count)):
            cls_idx = train_idx[train_labels == cls]
            if len(cls_idx) == 0: continue
            np.random.seed(SEED + fold)
            oversampled = np.random.choice(cls_idx, size=max_size, replace=True)
            oversampled_idx.extend(oversampled)
        
        oversampled_idx = np.array(oversampled_idx)
        
        # Train Model
        model = MultiviewGNN(input_dim, hidden_dim, output_dim, dropout=DROPOUT).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, factor=0.5)
        
        val_acc, test_pred = train_model(
            model, optimizer, scheduler, 
            torch.tensor(oversampled_idx).to(device), 
            torch.tensor(val_idx).to(device), 
            data, class_weights, epochs=300, patience=50, verbose=True
        )
        
        fold_val_accs.append(val_acc)
        fold_preds.append(test_pred)
        print(f"Fold {fold+1} Acc: {val_acc:.4f}")
    
    print(f"\nAverage CV Accuracy: {np.mean(fold_val_accs):.4f} ± {np.std(fold_val_accs):.4f}")
    
    # ---------------------------------------------------------
    # Phase 2: Final Training on Full Dataset + Validation Split
    # ---------------------------------------------------------
    print("\nTraining Final Model for Pseudo-Labeling...")
    
    # Prepare full training set with oversampling
    train_labels = y[idx_train_orig].cpu().numpy()
    class_count = np.bincount(train_labels)
    max_size = max(class_count)
    
    final_train_idx = []
    for cls in range(len(class_count)):
        cls_idx = idx_train_orig[train_labels == cls]
        if len(cls_idx) == 0: continue
        np.random.seed(SEED)
        oversampled = np.random.choice(cls_idx, size=max_size, replace=True)
        final_train_idx.extend(oversampled)
    
    final_train_idx = np.array(final_train_idx)
    
    # Create a random validation split (30%)
    np.random.seed(SEED)
    val_idx = np.random.choice(idx_train_orig, size=int(0.3 * len(idx_train_orig)), replace=False)
    train_idx = np.setdiff1d(final_train_idx, val_idx)
    
    final_model = MultiviewGNN(input_dim, hidden_dim, output_dim, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, factor=0.5)
    
    _, final_test_pred = train_model(
        final_model, optimizer, scheduler, 
        torch.tensor(train_idx).to(device), 
        torch.tensor(val_idx).to(device), 
        data, class_weights, epochs=300, patience=50, verbose=True
    )
    
    # ---------------------------------------------------------
    # Phase 3: Pseudo-Labeling & Ensemble
    # ---------------------------------------------------------
    print("\nExecuting Pseudo-Labeling Step...")
    
    # IMPORTANT: Use non-augmented data for inference (no noise)
    data_no_aug = Data(x=x, edge_index=edge_index, y=y)
    data_no_aug = data_no_aug.to(device)
    data_no_aug.test_idx = data.test_idx

    # Get high-confidence pseudo labels
    confident_idx, confident_labels = get_pseudo_labels(
        model=final_model, data=data_no_aug, 
        idx_test=data.test_idx, confidence_threshold=0.98
    )
    
    if len(confident_idx) > 0:
        print(f"Added {len(confident_idx)} high-confidence pseudo-labeled samples.")
        
        # Update labels in data object
        data.y[confident_idx] = confident_labels
        
        # Verify pseudo-label performance via CV
        evaluate_pseudo_model_with_kfold(
            data, idx_train_orig, confident_idx.cpu().numpy(), confident_labels,
            input_dim, hidden_dim, output_dim, class_weights, seed=SEED
        )
        
        # Retrain with combined data
        print("\nRetraining with Pseudo-Labels...")
        combined_train_idx = np.concatenate([final_train_idx, confident_idx.cpu().numpy()])
        
        np.random.seed(SEED + 1) # Vary seed for diversity
        val_idx = np.random.choice(idx_train_orig, size=int(0.3 * len(idx_train_orig)), replace=False)
        train_idx = np.setdiff1d(combined_train_idx, val_idx)
        
        pseudo_model = MultiviewGNN(input_dim, hidden_dim, output_dim, dropout=DROPOUT).to(device)
        optimizer = torch.optim.AdamW(pseudo_model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, factor=0.5)
        
        _, pseudo_test_pred = train_model(
            pseudo_model, optimizer, scheduler,
            torch.tensor(train_idx).to(device),
            torch.tensor(val_idx).to(device),
            data, class_weights, epochs=300, patience=50, verbose=True
        )
        
        # --- Ensemble Voting ---
        print("Performing Ensemble Voting...")
        # 1. Predictions from initial K-Folds
        fold_preds_array = np.array(fold_preds)
        fold_ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=fold_preds_array)
        
        # 2. Combine Pseudo-model pred + Fold Ensemble pred
        combined_preds = np.vstack([pseudo_test_pred, fold_ensemble_pred])
        
        # Majority voting (argmax of bincount)
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=combined_preds
        )
    else:
        print("No confident pseudo-labels found. Using Single Model Prediction.")
        final_predictions = final_test_pred
    
    # ---------------------------------------------------------
    # Phase 4: Submission
    # ---------------------------------------------------------
    team_name = "CONMEN"
    model_name = "MultiviewGNN"
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'{team_name}_{model_name}_{current_timestamp}_submission.txt'
    
    output_dir = f'{base_path}/output'
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    
    np.savetxt(file_path, final_predictions, fmt='%d')
    print(f"Submission saved to: {file_path} (Count: {len(final_predictions)})")
    
if __name__ == "__main__":
    main()