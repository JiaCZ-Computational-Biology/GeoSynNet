import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import global_max_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


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


def mse_loss(pred, true):
    return F.mse_loss(pred, true)


def kl_loss(latent):
    mean = torch.mean(latent, dim=0)
    var = torch.var(latent, dim=0)
    kl_div = -0.5 * torch.sum(1 + torch.log(var + 1e-10) - mean.pow(2) - var)
    return kl_div


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
        """
        h: node features [N, in_node_features]
        coords: node coordinates [N, 3]
        edge_index: [2, E]
        """
        row, col = edge_index

        # Compute relative distances
        coord_diff = coords[row] - coords[col]  # [E, 3]
        radial = torch.sum(coord_diff ** 2, dim=1, keepdim=True)  # [E, 1]

        # Edge features
        edge_feat = torch.cat([h[row], h[col], radial], dim=1)  # [E, 2*in_node_features + 1]
        edge_msg = self.edge_mlp(edge_feat)  # [E, hidden_features]

        # Update coordinates
        coord_weights = self.coord_mlp(edge_msg)  # [E, 1]
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
        h_updated = self.node_mlp(node_feat) + h[:, :self.out_node_features]  # residual connection

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


# Load data
train_csv_file = 'train_data.csv'
test_csv_file = 'validation_data.csv'
train_df = pd.read_csv(train_csv_file)
test_df = pd.read_csv(test_csv_file)
smiles_column = 'Smiles'
target_column = 'pchembl'

y_train = train_df[target_column].values.reshape(-1, 1)
y_test = test_df[target_column].values.reshape(-1, 1)

scaler = StandardScaler()
y_train_normalized = scaler.fit_transform(y_train)
y_test_normalized = scaler.transform(y_test)

train_df['pchembl_normalized'] = y_train_normalized
test_df['pchembl_normalized'] = y_test_normalized

train_df['pchembl_original'] = train_df[target_column]
test_df['pchembl_original'] = test_df[target_column]

# Prepare training data
train_data_list = []
for index, row in train_df.iterrows():
    smiles = str(row[smiles_column])
    try:
        atom_features, edge_index, coords = smiles_to_graph(smiles)
        ecfp = get_ecfp(smiles)
        data = Data(x=atom_features, edge_index=edge_index, coords=coords)
        data.y = torch.tensor(row['pchembl_normalized'], dtype=torch.float)
        data.y_original = torch.tensor(row['pchembl_original'], dtype=torch.float)
        train_data_list.append((data, ecfp))
    except Exception as e:
        print(f"Error processing training SMILES {smiles}: {e}")

# Prepare test data
test_data_list = []
for index, row in test_df.iterrows():
    smiles = str(row[smiles_column])
    try:
        atom_features, edge_index, coords = smiles_to_graph(smiles)
        ecfp = get_ecfp(smiles)
        data = Data(x=atom_features, edge_index=edge_index, coords=coords)
        data.y = torch.tensor(row['pchembl_normalized'], dtype=torch.float)
        data.y_original = torch.tensor(row['pchembl_original'], dtype=torch.float)
        test_data_list.append((data, ecfp))
    except Exception as e:
        print(f"Error processing test SMILES {smiles}: {e}")

train_loader = DataLoader(train_data_list, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)

# Initialize models - replaced GAT_GraphSAGE with EGNN
egnn_model = EGNN(n_output=1, num_features_xd=35, hidden_dim=128, output_dim=128, num_layers=3, dropout=0.3)
cnn_model = CNNNet(input_dim=1024, output_dim=1024)
combined_model = CombinedNet(input_dim=1025, hidden_dim=512, output_dim=1)

optimizer = torch.optim.Adam(
    list(egnn_model.parameters()) +
    list(cnn_model.parameters()) +
    list(combined_model.parameters()),
    lr=0.001,
    weight_decay=1e-4
)

best_mse = float('inf')
best_original_mse = float('inf')
best_model_file = 'best_model_egnn.pth'
lambda_kl = 0.001

# Training loop
for epoch in range(1000):
    egnn_model.train()
    cnn_model.train()
    combined_model.train()
    epoch_train_loss = 0.0
    num_batches = 0

    for batch_data, batch_ecfp in train_loader:
        optimizer.zero_grad()
        egnn_output = egnn_model(batch_data)
        cnn_output = cnn_model(batch_ecfp)
        combined_output = torch.cat((egnn_output, cnn_output), dim=1)
        final_output = combined_model(combined_output)

        batch_targets = batch_data.y.view(-1, 1)
        l_mse = mse_loss(final_output, batch_targets)
        l_kl = kl_loss(combined_output)
        total_loss = l_mse + lambda_kl * l_kl

        total_loss.backward()
        optimizer.step()

        epoch_train_loss += total_loss.item()
        num_batches += 1

    avg_train_loss = epoch_train_loss / num_batches

    # Evaluation
    egnn_model.eval()
    cnn_model.eval()
    combined_model.eval()
    test_mse_total = 0.0

    # Collect all predictions and targets for metric calculation
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_data, batch_ecfp in test_loader:
            egnn_output = egnn_model(batch_data)
            cnn_output = cnn_model(batch_ecfp)
            combined_output = torch.cat((egnn_output, cnn_output), dim=1)
            final_output = combined_model(combined_output)

            batch_targets = batch_data.y.view(-1, 1)
            mse = mse_loss(final_output, batch_targets)
            test_mse_total += mse.item()

            # Denormalize predictions
            denormalized_preds = scaler.inverse_transform(final_output.cpu().numpy())
            original_targets = batch_data.y_original.view(-1, 1).cpu().numpy()

            all_predictions.extend(denormalized_preds.flatten())
            all_targets.extend(original_targets.flatten())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate metrics on original scale
    test_mse_avg = test_mse_total / len(test_loader)
    test_original_mse = mean_squared_error(all_targets, all_predictions)
    test_mae = mean_absolute_error(all_targets, all_predictions)
    test_r2 = r2_score(all_targets, all_predictions)
    test_pearson, _ = pearsonr(all_targets, all_predictions)

    print(
        f"Epoch {epoch + 1:4d} | Train Loss: {avg_train_loss:.4f} | "
        f"Test MSE (Normalized): {test_mse_avg:.4f} | "
        f"Test MSE (Original): {test_original_mse:.4f} | "
        f"Test MAE: {test_mae:.4f} | "
        f"Test Pearson: {test_pearson:.4f} | "
        f"Test R²: {test_r2:.4f}"
    )

    if test_original_mse < best_original_mse:
        best_mse = test_mse_avg
        best_original_mse = test_original_mse
        torch.save({
            'egnn_model_state_dict': egnn_model.state_dict(),
            'cnn_model_state_dict': cnn_model.state_dict(),
            'combined_model_state_dict': combined_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'normalized_mse': test_mse_avg,
            'original_mse': test_original_mse,
            'mae': test_mae,
            'pearson': test_pearson,
            'r2': test_r2,
            'scaler': scaler,
        }, best_model_file)
        print(
            f"*** New best model saved at epoch {epoch + 1}, "
            f"MSE: {best_original_mse:.4f}, MAE: {test_mae:.4f}, "
            f"Pearson: {test_pearson:.4f}, R²: {test_r2:.4f} ***"
        )

print(f"\nTraining completed, Best Original MSE: {best_original_mse:.4f}")