import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import global_max_pool, global_mean_pool
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

    # Generate 3D coordinates for PaiNN
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


class RBFExpansion(nn.Module):
    """Radial Basis Function expansion for distance encoding"""

    def __init__(self, num_rbf=20, cutoff=10.0):
        super(RBFExpansion, self).__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        # Initialize centers and gamma values
        centers = torch.linspace(0, cutoff, num_rbf)
        self.register_buffer('centers', centers)
        gamma = 10.0 / cutoff
        self.register_buffer('gamma', torch.tensor(gamma))

    def forward(self, distances):
        """
        distances: [E, 1] edge distances
        returns: [E, num_rbf] RBF features
        """
        distances = distances.squeeze(-1)  # [E]
        # Apply cutoff
        cutoff_values = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        cutoff_values = cutoff_values * (distances < self.cutoff).float()

        # RBF expansion
        rbf = cutoff_values.unsqueeze(-1) * torch.exp(
            -self.gamma * (distances.unsqueeze(-1) - self.centers) ** 2
        )
        return rbf


class PaiNNMessage(nn.Module):
    """PaiNN message passing layer"""

    def __init__(self, hidden_dim, num_rbf):
        super(PaiNNMessage, self).__init__()
        self.hidden_dim = hidden_dim

        # Scalar message function
        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )

        # Filter generation network
        self.filter_network = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )

    def forward(self, s, v, edge_index, edge_rbf, edge_vec):
        """
        s: scalar features [N, hidden_dim]
        v: vector features [N, hidden_dim, 3]
        edge_index: [2, E]
        edge_rbf: [E, num_rbf]
        edge_vec: [E, 3] normalized edge vectors
        """
        row, col = edge_index

        # Generate filter values
        W = self.filter_network(edge_rbf)  # [E, hidden_dim * 3]
        W = W.view(-1, self.hidden_dim, 3)  # [E, hidden_dim, 3]

        # Split for different transformations
        W_s = W[:, :, 0]  # [E, hidden_dim]
        W_v1 = W[:, :, 1]  # [E, hidden_dim]
        W_v2 = W[:, :, 2]  # [E, hidden_dim]

        # Scalar messages
        s_message = self.scalar_message_mlp(s[col])  # [E, hidden_dim * 3]
        s_message = s_message.view(-1, self.hidden_dim, 3)  # [E, hidden_dim, 3]

        ds = s_message[:, :, 0]  # [E, hidden_dim]
        dv1 = s_message[:, :, 1]  # [E, hidden_dim]
        dv2 = s_message[:, :, 2]  # [E, hidden_dim]

        # Apply filters
        ds = ds * W_s  # [E, hidden_dim]

        # Vector messages
        # v[col]: [E, hidden_dim, 3]
        dv = v[col] * W_v1.unsqueeze(-1) + dv1.unsqueeze(-1) * edge_vec.unsqueeze(1)  # [E, hidden_dim, 3]

        # Scalar contribution from vectors
        ds_from_v = torch.sum(v[col] * edge_vec.unsqueeze(1), dim=-1) * W_v2  # [E, hidden_dim]
        ds = ds + ds_from_v

        # Aggregate messages
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

        # Update networks
        self.update_U = nn.Linear(hidden_dim, hidden_dim)
        self.update_V = nn.Linear(hidden_dim, hidden_dim)

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )

    def forward(self, s, v):
        """
        s: scalar features [N, hidden_dim]
        v: vector features [N, hidden_dim, 3]
        """
        # Vector norm
        v_norm = torch.sqrt(torch.sum(v ** 2, dim=-1) + 1e-8)  # [N, hidden_dim]

        # Combine scalar and vector information
        U_v = self.update_U(v_norm)  # [N, hidden_dim]
        V_v = self.update_V(v_norm)  # [N, hidden_dim]

        combined = torch.cat([s, V_v], dim=-1)  # [N, hidden_dim * 2]
        update_values = self.update_mlp(combined)  # [N, hidden_dim * 3]
        update_values = update_values.view(-1, self.hidden_dim, 3)

        # Update scalar features
        a_ss = update_values[:, :, 0]  # [N, hidden_dim]
        a_sv = update_values[:, :, 1]  # [N, hidden_dim]
        a_vv = update_values[:, :, 2]  # [N, hidden_dim]

        # Residual updates
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

        # Initial embeddings
        self.node_embedding = nn.Linear(num_features_xd, hidden_dim)

        # RBF expansion for distances
        self.rbf_expansion = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)

        # PaiNN layers
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.message_layers.append(PaiNNMessage(hidden_dim, num_rbf))
            self.update_layers.append(PaiNNUpdate(hidden_dim))

        # Output network
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, coords, batch = data.x, data.edge_index, data.coords, data.batch

        # Initialize scalar and vector features
        s = self.node_embedding(x)  # [N, hidden_dim]
        v = torch.zeros(s.size(0), self.hidden_dim, 3, device=s.device)  # [N, hidden_dim, 3]

        # Compute edge features
        row, col = edge_index
        edge_vec = coords[row] - coords[col]  # [E, 3]
        edge_dist = torch.sqrt(torch.sum(edge_vec ** 2, dim=-1, keepdim=True) + 1e-8)  # [E, 1]
        edge_vec_normalized = edge_vec / (edge_dist + 1e-8)  # [E, 3]

        # RBF expansion of distances
        edge_rbf = self.rbf_expansion(edge_dist)  # [E, num_rbf]

        # Message passing and updates
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            # Message passing
            ds, dv = message_layer(s, v, edge_index, edge_rbf, edge_vec_normalized)
            s = s + ds
            v = v + dv

            # Update
            s, v = update_layer(s, v)
            s = self.dropout(s)

        # Global pooling (only use scalar features for output)
        s_pooled = global_mean_pool(s, batch)  # [batch_size, hidden_dim]

        # Output
        output = self.output_network(s_pooled)  # [batch_size, output_dim]

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

# Initialize models - PaiNN and CNN both output 128 dimensions
painn_model = PaiNN(num_features_xd=35, hidden_dim=128, output_dim=128,
                    num_layers=3, num_rbf=20, cutoff=10.0, dropout=0.3)
cnn_model = CNNNet(input_dim=1024, output_dim=128, dropout=0.3)
combined_model = CombinedNet(input_dim=256, hidden_dim=512, output_dim=1)  # 128 + 128 = 256

optimizer = torch.optim.Adam(
    list(painn_model.parameters()) +
    list(cnn_model.parameters()) +
    list(combined_model.parameters()),
    lr=0.001,
    weight_decay=1e-4
)

best_mse = float('inf')
best_original_mse = float('inf')
best_model_file = 'best_model_painn.pth'
lambda_kl = 0.001

# Training loop
for epoch in range(1000):
    painn_model.train()
    cnn_model.train()
    combined_model.train()
    epoch_train_loss = 0.0
    num_batches = 0

    for batch_data, batch_ecfp in train_loader:
        optimizer.zero_grad()
        painn_output = painn_model(batch_data)  # [batch_size, 128]
        cnn_output = cnn_model(batch_ecfp)  # [batch_size, 128]
        combined_output = torch.cat((painn_output, cnn_output), dim=1)  # [batch_size, 256]
        final_output = combined_model(combined_output)  # [batch_size, 1]

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
    painn_model.eval()
    cnn_model.eval()
    combined_model.eval()
    test_mse_total = 0.0

    # Collect all predictions and targets for metric calculation
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_data, batch_ecfp in test_loader:
            painn_output = painn_model(batch_data)
            cnn_output = cnn_model(batch_ecfp)
            combined_output = torch.cat((painn_output, cnn_output), dim=1)
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
            'painn_model_state_dict': painn_model.state_dict(),
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