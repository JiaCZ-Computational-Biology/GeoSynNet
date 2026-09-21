import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import global_max_pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# Set random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


# ==================== Helper functions ====================
def one_of_k_encoding_unk(x, valid_entries):
    if x not in valid_entries:
        x = 'Unknown'
    return [1 if entry == x else 0 for entry in valid_entries]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Generate 3D coordinates for EGNN
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

        # Get 3D coordinates
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


# ==================== Model definitions ====================
class EGNNLayer(nn.Module):
    """E(n) Equivariant Graph Neural Network Layer"""

    def __init__(self, in_node_features, hidden_features, out_node_features):
        super(EGNNLayer, self).__init__()

        self.in_node_features = in_node_features
        self.hidden_features = hidden_features
        self.out_node_features = out_node_features

        # Edge model: phi_e
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_node_features + 1, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU()
        )

        # Node model: phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_node_features + hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_node_features)
        )

        # Coordinate model: phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1, bias=False)
        )

    def forward(self, h, coords, edge_index):
        row, col = edge_index

        # Compute relative distances
        coord_diff = coords[row] - coords[col]
        radial = torch.sum(coord_diff ** 2, dim=1, keepdim=True)

        # Edge features
        edge_feat = torch.cat([h[row], h[col], radial], dim=1)
        edge_msg = self.edge_mlp(edge_feat)

        # Update coordinates
        coord_weights = self.coord_mlp(edge_msg)
        coord_diff_normalized = coord_diff / (torch.sqrt(radial) + 1e-8)
        coord_update = coord_diff_normalized * coord_weights

        # Aggregate coordinate updates
        coords_updated = coords.clone()
        coords_updated.index_add_(0, row, coord_update)

        # Aggregate edge messages for node updates
        agg_msg = torch.zeros(h.size(0), self.hidden_features, device=h.device)
        agg_msg.index_add_(0, row, edge_msg)

        # Update node features
        node_feat = torch.cat([h, agg_msg], dim=1)
        h_updated = self.node_mlp(node_feat) + h[:, :self.out_node_features]

        return h_updated, coords_updated


class EGNN(nn.Module):
    """E(n) Equivariant Graph Neural Network for molecular property prediction"""

    def __init__(self, n_output=1, num_features_xd=35, hidden_dim=128, output_dim=128, num_layers=3, dropout=0.3):
        super(EGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Initial node embedding
        self.node_embedding = nn.Linear(num_features_xd, hidden_dim)

        # EGNN layers
        self.egnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.egnn_layers.append(
                EGNNLayer(hidden_dim, hidden_dim, hidden_dim)
            )

        # Output layers
        self.fc_g1 = nn.Linear(hidden_dim, 1500)
        self.fc_g2 = nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(output_dim, n_output)

    def forward(self, data):
        x, edge_index, coords, batch = data.x, data.edge_index, data.coords, data.batch

        # Initial embedding
        h = self.relu(self.node_embedding(x))

        # EGNN layers
        for egnn_layer in self.egnn_layers:
            h, coords = egnn_layer(h, coords, edge_index)
            h = self.dropout(h)

        # Global pooling
        x = global_max_pool(h, batch)

        # Output
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        out = self.out(x)

        return out


class CNNNet(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(128 * input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CombinedNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ==================== Main evaluation code ====================
def evaluate_model(model_path, test_csv_file, smiles_column='Smiles', target_column='pchembl', device='cpu'):
    """
    Load the best model and evaluate it on an independent test set
    """

    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load saved model
    print(f"\nLoading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Initialize models
    egnn_model = EGNN(n_output=1, num_features_xd=35, hidden_dim=128, output_dim=128, num_layers=3, dropout=0.3).to(
        device)
    cnn_model = CNNNet(input_dim=1024, output_dim=1024).to(device)
    combined_model = CombinedNet(input_dim=1025, hidden_dim=512, output_dim=1).to(device)

    # Load model parameters
    egnn_model.load_state_dict(checkpoint['egnn_model_state_dict'])
    cnn_model.load_state_dict(checkpoint['cnn_model_state_dict'])
    combined_model.load_state_dict(checkpoint['combined_model_state_dict'])

    # Load scaler
    scaler = checkpoint['scaler']

    # Set to evaluation mode
    egnn_model.eval()
    cnn_model.eval()
    combined_model.eval()

    print(f"Model loaded successfully!")
    print(f"Best Normalized MSE during training: {checkpoint['normalized_mse']:.4f}")
    print(f"Best Original MSE during training: {checkpoint['original_mse']:.4f}")

    # Load test data
    print(f"\nLoading test data: {test_csv_file}")
    test_df = pd.read_csv(test_csv_file)
    print(f"Number of test samples: {len(test_df)}")

    # Prepare test data
    test_data_list = []
    failed_count = 0

    for index, row in test_df.iterrows():
        smiles = str(row[smiles_column])
        try:
            atom_features, edge_index, coords = smiles_to_graph(smiles)
            ecfp = get_ecfp(smiles)
            data = Data(x=atom_features, edge_index=edge_index, coords=coords)
            data.y_original = torch.tensor(row[target_column], dtype=torch.float)
            test_data_list.append((data, ecfp))
        except Exception as e:
            failed_count += 1
            print(f"Error processing SMILES {smiles}: {e}")

    print(f"Successfully processed: {len(test_data_list)} samples, failed: {failed_count} samples")

    if len(test_data_list) == 0:
        print("No valid test samples!")
        return

    # Create data loader
    test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)

    # Prediction
    print("\nStarting prediction...")
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_data, batch_ecfp in test_loader:
            # Move data to device
            batch_data = batch_data.to(device)
            batch_ecfp = batch_ecfp.to(device)

            # Forward propagation
            egnn_output = egnn_model(batch_data)
            cnn_output = cnn_model(batch_ecfp)
            combined_output = torch.cat((egnn_output, cnn_output), dim=1)
            final_output = combined_model(combined_output)

            # Denormalize predictions
            denormalized_preds = scaler.inverse_transform(final_output.cpu().numpy())
            original_targets = batch_data.y_original.cpu().numpy()

            all_predictions.extend(denormalized_preds.flatten())
            all_targets.extend(original_targets.flatten())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate evaluation metrics
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    pearson_corr, p_value = pearsonr(all_targets, all_predictions)

    # Print results
    print("\n" + "=" * 60)
    print("Independent Test Set Evaluation Results:")
    print("=" * 60)
    print(f"Number of samples:                {len(all_targets)}")
    print(f"MSE (Mean Squared Error):         {mse:.4f}")
    print(f"RMSE (Root Mean Squared Error):   {rmse:.4f}")
    print(f"MAE (Mean Absolute Error):        {mae:.4f}")
    print(f"R² (Coefficient of Determination): {r2:.4f}")
    print(f"Pearson correlation coefficient:  {pearson_corr:.4f}")
    print(f"P-value:                          {p_value:.4e}")
    print("=" * 60)

    # Save prediction results
    results_df = pd.DataFrame({
        'True_Value': all_targets,
        'Predicted_Value': all_predictions,
        'Absolute_Error': np.abs(all_targets - all_predictions),
        'Squared_Error': (all_targets - all_predictions) ** 2
    })

    output_file = 'test_predictions.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nPrediction results saved to: {output_file}")

    # Print statistical information
    print("\nPrediction statistics:")
    print(f"Minimum: {all_predictions.min():.4f}")
    print(f"Maximum: {all_predictions.max():.4f}")
    print(f"Mean: {all_predictions.mean():.4f}")
    print(f"Standard deviation: {all_predictions.std():.4f}")

    print("\nTrue value statistics:")
    print(f"Minimum: {all_targets.min():.4f}")
    print(f"Maximum: {all_targets.max():.4f}")
    print(f"Mean: {all_targets.mean():.4f}")
    print(f"Standard deviation: {all_targets.std():.4f}")

    # Error analysis
    print("\nError analysis:")
    print(f"Mean absolute error: {mae:.4f}")
    print(f"Maximum absolute error: {np.max(np.abs(all_targets - all_predictions)):.4f}")
    print(f"Minimum absolute error: {np.min(np.abs(all_targets - all_predictions)):.4f}")
    print(f"Error standard deviation: {np.std(all_targets - all_predictions):.4f}")

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr,
        'p_value': p_value,
        'predictions': all_predictions,
        'targets': all_targets
    }


# ==================== Usage example ====================
if __name__ == "__main__":
    # Configuration parameters
    model_path = 'best_model_egnn.pth'  # Best model path
    test_csv_file = 'test_data.csv'  # Independent test set path
    smiles_column = 'Smiles'  # SMILES column name
    target_column = 'pchembl'  # Target value column name
    device = 'cpu'  # Use 'cuda' if a GPU is available

    # Run evaluation
    results = evaluate_model(
        model_path=model_path,
        test_csv_file=test_csv_file,
        smiles_column=smiles_column,
        target_column=target_column,
        device=device
    )