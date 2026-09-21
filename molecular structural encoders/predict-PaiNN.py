import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr


def one_of_k_encoding_unk(x, valid_entries):
    if x not in valid_entries:
        x = 'Unknown'
    return [1 if entry == x else 0 for entry in valid_entries]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Generate 3D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)

    num_atoms = mol.GetNumAtoms()
    atom_features = []
    coords = []

    conformer = mol.GetConformer()
    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Unknown']) + \
                  one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2
                  ]) + [atom.GetIsAromatic()] + \
                  one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        atom_feats = np.array(results).astype(np.float32)
        atom_features.append(atom_feats)

        pos = conformer.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])

    atom_features = torch.tensor(atom_features, dtype=torch.float)
    coords = torch.tensor(coords, dtype=torch.float)
    adj_matrix = torch.zeros((num_atoms, num_atoms), dtype=torch.float)

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj_matrix[start, end] = 1.0
        adj_matrix[end, start] = 1.0

    edge_index = adj_matrix.nonzero(as_tuple=False).t().long()
    return atom_features, edge_index, coords


def get_ecfp(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot generate molecule from SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return torch.tensor(np.array(fp), dtype=torch.float).view(1, -1)


# ==================== Custom Dataset ====================

class MoleculeDataset(Dataset):
    """Custom dataset for molecule data"""

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_fn(batch):
    """Custom collate function for batching"""
    graph_data_list = []
    ecfp_list = []
    smiles_list = []

    for item in batch:
        graph_data_list.append(item[0])
        ecfp_list.append(item[1])
        smiles_list.append(item[2])

    batch_data = Batch.from_data_list(graph_data_list)
    batch_ecfp = torch.cat(ecfp_list, dim=0)

    return batch_data, batch_ecfp, smiles_list


# ==================== Model Definitions ====================

class RBFExpansion(nn.Module):
    """Radial Basis Function expansion for distance encoding"""

    def __init__(self, num_rbf=20, cutoff=10.0):
        super(RBFExpansion, self).__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        centers = torch.linspace(0, cutoff, num_rbf)
        self.register_buffer('centers', centers)
        gamma = 10.0 / cutoff
        self.register_buffer('gamma', torch.tensor(gamma))

    def forward(self, distances):
        distances = distances.squeeze(-1)
        cutoff_values = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        cutoff_values = cutoff_values * (distances < self.cutoff).float()

        rbf = cutoff_values.unsqueeze(-1) * torch.exp(
            -self.gamma * (distances.unsqueeze(-1) - self.centers) ** 2
        )
        return rbf


class PaiNNMessage(nn.Module):
    """PaiNN message passing layer"""

    def __init__(self, hidden_dim, num_rbf):
        super(PaiNNMessage, self).__init__()
        self.hidden_dim = hidden_dim

        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )

        self.filter_network = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )

    def forward(self, s, v, edge_index, edge_rbf, edge_vec):
        row, col = edge_index

        W = self.filter_network(edge_rbf)
        W = W.view(-1, self.hidden_dim, 3)

        W_s = W[:, :, 0]
        W_v1 = W[:, :, 1]
        W_v2 = W[:, :, 2]

        s_message = self.scalar_message_mlp(s[col])
        s_message = s_message.view(-1, self.hidden_dim, 3)

        ds = s_message[:, :, 0]
        dv1 = s_message[:, :, 1]
        dv2 = s_message[:, :, 2]

        ds = ds * W_s
        dv = v[col] * W_v1.unsqueeze(-1) + dv1.unsqueeze(-1) * edge_vec.unsqueeze(1)
        ds_from_v = torch.sum(v[col] * edge_vec.unsqueeze(1), dim=-1) * W_v2
        ds = ds + ds_from_v

        s_updated = torch.zeros_like(s)
        s_updated.index_add_(0, row, ds)

        v_updated = torch.zeros_like(v)
        v_updated.index_add_(0, row, dv)

        return s_updated, v_updated


class PaiNNUpdate(nn.Module):
    """PaiNN update layer for scalar and vector features"""

    def __init__(self, hidden_dim):
        super(PaiNNUpdate, self).__init__()
        self.hidden_dim = hidden_dim

        self.update_U = nn.Linear(hidden_dim, hidden_dim)
        self.update_V = nn.Linear(hidden_dim, hidden_dim)

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )

    def forward(self, s, v):
        v_norm = torch.sqrt(torch.sum(v ** 2, dim=-1) + 1e-8)

        U_v = self.update_U(v_norm)
        V_v = self.update_V(v_norm)

        combined = torch.cat([s, V_v], dim=-1)
        update_values = self.update_mlp(combined)
        update_values = update_values.view(-1, self.hidden_dim, 3)

        a_ss = update_values[:, :, 0]
        a_sv = update_values[:, :, 1]
        a_vv = update_values[:, :, 2]

        ds = a_ss + a_sv * U_v
        dv = a_vv.unsqueeze(-1) * v

        return s + ds, v + dv


class PaiNN(nn.Module):
    """Polarizable Atom Interaction Neural Network"""

    def __init__(self, num_features_xd=35, hidden_dim=128, output_dim=128,
                 num_layers=3, num_rbf=20, cutoff=10.0, dropout=0.3):
        super(PaiNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.node_embedding = nn.Linear(num_features_xd, hidden_dim)
        self.rbf_expansion = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.message_layers.append(PaiNNMessage(hidden_dim, num_rbf))
            self.update_layers.append(PaiNNUpdate(hidden_dim))

        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, coords, batch = data.x, data.edge_index, data.coords, data.batch

        s = self.node_embedding(x)
        v = torch.zeros(s.size(0), self.hidden_dim, 3, device=s.device)

        row, col = edge_index
        edge_vec = coords[row] - coords[col]
        edge_dist = torch.sqrt(torch.sum(edge_vec ** 2, dim=-1, keepdim=True) + 1e-8)
        edge_vec_normalized = edge_vec / (edge_dist + 1e-8)

        edge_rbf = self.rbf_expansion(edge_dist)

        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            ds, dv = message_layer(s, v, edge_index, edge_rbf, edge_vec_normalized)
            s = s + ds
            v = v + dv
            s, v = update_layer(s, v)
            s = self.dropout(s)

        s_pooled = global_mean_pool(s, batch)
        output = self.output_network(s_pooled)

        return output


class CNNNet(nn.Module):
    def __init__(self, input_dim, output_dim=128, dropout=0.3):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(128 * input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, ecfp):
        ecfp = ecfp.squeeze(1).unsqueeze(1)
        x = self.relu(self.conv1(ecfp))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CombinedNet(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=1):
        super(CombinedNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def evaluate_model(model_path, test_csv_file, smiles_column='Smiles', target_column='pchembl',
                   batch_size=64, device='cpu'):
    """
    Evaluate the saved model on independent test set

    Args:
        model_path: Path to saved model checkpoint
        test_csv_file: Path to test CSV file
        smiles_column: Name of SMILES column
        target_column: Name of target column
        batch_size: Batch size for evaluation
        device: Device to use ('cpu' or 'cuda')

    Returns:
        Dictionary containing all metrics and predictions
    """

    print(f"Loading model from {model_path}...")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get scaler
    scaler = checkpoint['scaler']

    # Initialize models
    painn_model = PaiNN(num_features_xd=35, hidden_dim=128, output_dim=128,
                        num_layers=3, num_rbf=20, cutoff=10.0, dropout=0.3)
    cnn_model = CNNNet(input_dim=1024, output_dim=128, dropout=0.3)
    combined_model = CombinedNet(input_dim=256, hidden_dim=512, output_dim=1)

    # Load state dicts
    painn_model.load_state_dict(checkpoint['painn_model_state_dict'])
    cnn_model.load_state_dict(checkpoint['cnn_model_state_dict'])
    combined_model.load_state_dict(checkpoint['combined_model_state_dict'])

    # Move to device
    painn_model = painn_model.to(device)
    cnn_model = cnn_model.to(device)
    combined_model = combined_model.to(device)

    # Set to evaluation mode
    painn_model.eval()
    cnn_model.eval()
    combined_model.eval()

    print(f"Loading test data from {test_csv_file}...")

    # Load test data
    test_df = pd.read_csv(test_csv_file)

    # Prepare test data
    test_data_list = []
    failed_smiles = []

    for index, row in test_df.iterrows():
        smiles = str(row[smiles_column])
        try:
            atom_features, edge_index, coords = smiles_to_graph(smiles)
            ecfp = get_ecfp(smiles)
            data = Data(x=atom_features, edge_index=edge_index, coords=coords)
            data.y_original = torch.tensor(row[target_column], dtype=torch.float)
            test_data_list.append((data, ecfp, smiles))
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            failed_smiles.append(smiles)

    print(f"Successfully processed {len(test_data_list)} molecules")
    print(f"Failed to process {len(failed_smiles)} molecules")

    # Create dataset and data loader
    test_dataset = MoleculeDataset(test_data_list)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False, collate_fn=collate_fn)

    # Collect predictions and targets
    all_predictions = []
    all_targets = []
    all_smiles = []

    print("Running predictions...")

    with torch.no_grad():
        for batch_idx, (batch_data, batch_ecfp, batch_smiles_list) in enumerate(test_loader):
            # Move to device
            batch_data = batch_data.to(device)
            batch_ecfp = batch_ecfp.to(device)

            # Forward pass
            painn_output = painn_model(batch_data)
            cnn_output = cnn_model(batch_ecfp)
            combined_output = torch.cat((painn_output, cnn_output), dim=1)
            final_output = combined_model(combined_output)

            # Denormalize predictions
            denormalized_preds = scaler.inverse_transform(final_output.cpu().numpy())
            original_targets = batch_data.y_original.view(-1, 1).cpu().numpy()

            all_predictions.extend(denormalized_preds.flatten())
            all_targets.extend(original_targets.flatten())
            all_smiles.extend(batch_smiles_list)

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * batch_size} / {len(test_data_list)} molecules")

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    pearson_corr, pearson_pvalue = pearsonr(all_targets, all_predictions)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS ON INDEPENDENT TEST SET")
    print("=" * 60)
    print(f"Number of test samples: {len(all_predictions)}")
    print(f"Failed samples: {len(failed_smiles)}")
    print("-" * 60)
    print(f"MSE (Mean Squared Error):        {mse:.4f}")
    print(f"RMSE (Root Mean Squared Error):  {rmse:.4f}")
    print(f"MAE (Mean Absolute Error):       {mae:.4f}")
    print(f"R² (R-squared):                  {r2:.4f}")
    print(f"Pearson Correlation:             {pearson_corr:.4f}")
    print(f"Pearson p-value:                 {pearson_pvalue:.4e}")
    print("=" * 60)

    # Calculate additional statistics
    residuals = all_targets - all_predictions
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    print("\nRESIDUAL STATISTICS")
    print("-" * 60)
    print(f"Mean Residual:                   {mean_residual:.4f}")
    print(f"Std Residual:                    {std_residual:.4f}")
    print(f"Min Residual:                    {np.min(residuals):.4f}")
    print(f"Max Residual:                    {np.max(residuals):.4f}")
    print("=" * 60)

    # Save results to CSV
    results_df = pd.DataFrame({
        'SMILES': all_smiles,
        'True_Value': all_targets,
        'Predicted_Value': all_predictions,
        'Residual': residuals,
        'Absolute_Error': np.abs(residuals)
    })

    output_file = 'independent_test_predictions.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")

    # Save metrics summary
    metrics_summary = {
        'n_samples': len(all_predictions),
        'n_failed': len(failed_smiles),
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Pearson_Correlation': pearson_corr,
        'Pearson_pvalue': pearson_pvalue,
        'Mean_Residual': mean_residual,
        'Std_Residual': std_residual,
        'Min_Residual': np.min(residuals),
        'Max_Residual': np.max(residuals)
    }

    metrics_df = pd.DataFrame([metrics_summary])
    metrics_file = 'independent_test_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics summary saved to {metrics_file}")

    # Return all results
    return {
        'metrics': metrics_summary,
        'predictions': all_predictions,
        'targets': all_targets,
        'smiles': all_smiles,
        'residuals': residuals,
        'failed_smiles': failed_smiles,
        'results_df': results_df
    }


# ==================== Main Execution ====================

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = 'best_model_painn.pth'  # Path to your saved model
    TEST_CSV = 'test_data.csv'  # Path to your independent test set
    SMILES_COLUMN = 'Smiles'  # Name of SMILES column in CSV
    TARGET_COLUMN = 'pchembl'  # Name of target column in CSV
    BATCH_SIZE = 64
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")

    # Run evaluation
    results = evaluate_model(
        model_path=MODEL_PATH,
        test_csv_file=TEST_CSV,
        smiles_column=SMILES_COLUMN,
        target_column=TARGET_COLUMN,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    # Optional: Create visualization
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))

        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(results['targets'], results['predictions'], alpha=0.5, s=20)
        plt.plot([results['targets'].min(), results['targets'].max()],
                 [results['targets'].min(), results['targets'].max()],
                 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(f'Prediction vs True Values\nR² = {results["metrics"]["R2"]:.4f}, '
                  f'Pearson = {results["metrics"]["Pearson_Correlation"]:.4f}', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Residual plot
        plt.subplot(1, 2, 2)
        plt.scatter(results['predictions'], results['residuals'], alpha=0.5, s=20)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Values', fontsize=12)
        plt.ylabel('Residuals', fontsize=12)
        plt.title(f'Residual Plot\nMAE = {results["metrics"]["MAE"]:.4f}, '
                  f'RMSE = {results["metrics"]["RMSE"]:.4f}', fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('independent_test_evaluation.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to 'independent_test_evaluation.png'")

    except ImportError:
        print("\nMatplotlib not available. Skipping visualization.")

    print("\nEvaluation completed successfully!")