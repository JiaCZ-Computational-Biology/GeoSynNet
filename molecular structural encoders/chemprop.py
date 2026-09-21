import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import global_mean_pool, global_add_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def one_of_k_encoding(x, allowable_set):
    """One-hot encoding with unknown category"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]


def atom_features(atom):
    """Generate atom features compatible with Chemprop - 133 dimensions"""
    features = (
            one_of_k_encoding(atom.GetSymbol(),
                              ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                               'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd',
                               'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
                               'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +  # 44 dimensions
            one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +  # 11 dimensions
            one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +  # 11 dimensions
            one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +  # 11 dimensions
            [atom.GetIsAromatic()] +  # 1 dimension
            one_of_k_encoding(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.S,
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
                'other']) +  # 7 dimensions
            [atom.GetFormalCharge()] +  # 1 dimension
            [atom.GetNumRadicalElectrons()] +  # 1 dimension
            one_of_k_encoding(atom.GetChiralTag(), [
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_OTHER]) +  # 4 dimensions
            [atom.GetMass() * 0.01] +  # 1 dimension
            [atom.IsInRing()] +  # 1 dimension
            one_of_k_encoding(atom.GetTotalValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8]) +  # 9 dimensions
            [atom.IsInRingSize(3)] +  # 1 dimension
            [atom.IsInRingSize(4)] +  # 1 dimension
            [atom.IsInRingSize(5)] +  # 1 dimension
            [atom.IsInRingSize(6)] +  # 1 dimension
            [atom.IsInRingSize(7)] +  # 1 dimension
            [atom.IsInRingSize(8)] +  # 1 dimension
            [int(atom.GetIsAromatic() and atom.IsInRingSize(5))] +  # 1 dimension
            [int(atom.GetIsAromatic() and atom.IsInRingSize(6))] +  # 1 dimension
            one_of_k_encoding(len([x for x in atom.GetNeighbors() if x.GetIsAromatic()]),
                              [0, 1, 2, 3, 4, 5]) +  # 6 dimensions
            [atom.GetTotalDegree()] +  # 1 dimension
            [int(atom.GetSymbol() in ['C', 'N', 'O', 'S'])] +  # 1 dimension
            [int(atom.GetSymbol() in ['F', 'Cl', 'Br', 'I'])] +  # 1 dimension
            [0] * 14  # 14-dimensional padding
    )
    # Total: 44+11+11+11+1+7+1+1+4+1+1+9+6+2+6+1+1+1+14 = 133 dimensions
    return np.array(features).astype(np.float32)


def bond_features(bond):
    """Generate bond features compatible with Chemprop"""
    if bond is None:
        fbond = [1, 0, 0, 0, 0, 0]
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
        ]
    return np.array(fbond).astype(np.float32)


def smiles_to_graph(smiles):
    """Convert SMILES to molecular graph compatible with Chemprop"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    num_atoms = mol.GetNumAtoms()
    atom_features_list = []

    for atom in mol.GetAtoms():
        atom_features_list.append(atom_features(atom))

    x = torch.tensor(atom_features_list, dtype=torch.float)

    # Create edge index and edge features for directed graph (both directions)
    edge_indices = []
    edge_features_list = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Add both directions
        edge_indices.append([i, j])
        edge_indices.append([j, i])

        bond_feat = bond_features(bond)
        edge_features_list.append(bond_feat)
        edge_features_list.append(bond_feat)

    # If there are no bonds, create empty tensors
    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features_list, dtype=torch.float)

    return x, edge_index, edge_attr


def get_ecfp(smiles, radius=2, nBits=1024):
    """Generate ECFP fingerprint"""
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


class MPNEncoder(nn.Module):
    """Message Passing Neural Network Encoder (Chemprop-style)"""

    def __init__(self, atom_fdim=133, bond_fdim=6, hidden_size=300, depth=3, dropout=0.0):
        super(MPNEncoder, self).__init__()

        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = nn.Dropout(dropout)

        # Learnable transformation for initial message
        self.W_i = nn.Linear(atom_fdim + bond_fdim, hidden_size, bias=False)

        # Message passing layers
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(atom_fdim + hidden_size, hidden_size)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Get number of nodes
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            # Handle molecules with no bonds
            node_repr = self.W_o(x)
            return global_mean_pool(node_repr, batch)

        # Initialize messages: concatenate target atom features with bond features
        row, col = edge_index
        message_input = torch.cat([x[col], edge_attr], dim=1)  # [num_edges, atom_fdim + bond_fdim]
        messages = F.relu(self.W_i(message_input))  # [num_edges, hidden_size]

        # Message passing
        for _ in range(self.depth - 1):
            # Aggregate messages from incoming edges
            nei_messages = torch.zeros(num_nodes, self.hidden_size, device=x.device)
            nei_messages.index_add_(0, row, messages)

            # Get neighbor messages for each edge (exclude reverse edge)
            a2nei = nei_messages[col] - messages  # [num_edges, hidden_size]

            # Update messages
            messages = F.relu(self.W_h(a2nei))  # [num_edges, hidden_size]
            messages = self.dropout(messages)

        # Aggregate final messages to atoms
        atom_messages = torch.zeros(num_nodes, self.hidden_size, device=x.device)
        atom_messages.index_add_(0, row, messages)

        # Combine with atom features
        atom_hiddens = torch.cat([x, atom_messages], dim=1)  # [num_nodes, atom_fdim + hidden_size]
        atom_hiddens = F.relu(self.W_o(atom_hiddens))  # [num_nodes, hidden_size]
        atom_hiddens = self.dropout(atom_hiddens)

        # Global pooling
        mol_repr = global_mean_pool(atom_hiddens, batch)  # [batch_size, hidden_size]

        return mol_repr


class ChempropModel(nn.Module):
    """Chemprop model with FFN for property prediction"""

    def __init__(self, atom_fdim=133, bond_fdim=6, hidden_size=300, depth=3,
                 ffn_hidden_size=300, ffn_num_layers=2, output_dim=128, dropout=0.0):
        super(ChempropModel, self).__init__()

        # Message Passing Network
        self.mpn = MPNEncoder(atom_fdim, bond_fdim, hidden_size, depth, dropout)

        # Feed-forward network
        first_linear_dim = hidden_size

        ffn_layers = []
        for i in range(ffn_num_layers):
            if i == 0:
                ffn_layers.append(nn.Linear(first_linear_dim, ffn_hidden_size))
            else:
                ffn_layers.append(nn.Linear(ffn_hidden_size, ffn_hidden_size))
            ffn_layers.append(nn.ReLU())
            ffn_layers.append(nn.Dropout(dropout))

        # Final output layer
        ffn_layers.append(nn.Linear(ffn_hidden_size, output_dim))

        self.ffn = nn.Sequential(*ffn_layers)

    def forward(self, data):
        # Get molecular representation from MPN
        mol_repr = self.mpn(data)

        # Pass through FFN
        output = self.ffn(mol_repr)

        return output


class CNNNet(nn.Module):
    """CNN for ECFP fingerprint processing"""

    def __init__(self, input_dim, output_dim=128, dropout=0.3):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
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
    """Combined network for final prediction"""

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
        atom_features_tensor, edge_index, edge_attr = smiles_to_graph(smiles)
        ecfp = get_ecfp(smiles)
        data = Data(x=atom_features_tensor, edge_index=edge_index, edge_attr=edge_attr)
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
        atom_features_tensor, edge_index, edge_attr = smiles_to_graph(smiles)
        ecfp = get_ecfp(smiles)
        data = Data(x=atom_features_tensor, edge_index=edge_index, edge_attr=edge_attr)
        data.y = torch.tensor(row['pchembl_normalized'], dtype=torch.float)
        data.y_original = torch.tensor(row['pchembl_original'], dtype=torch.float)
        test_data_list.append((data, ecfp))
    except Exception as e:
        print(f"Error processing test SMILES {smiles}: {e}")

train_loader = DataLoader(train_data_list, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)

# Initialize models - Chemprop and CNN both output 128 dimensions
chemprop_model = ChempropModel(
    atom_fdim=133,  # 133-dimensional atom features
    bond_fdim=6,
    hidden_size=300,
    depth=3,
    ffn_hidden_size=300,
    ffn_num_layers=2,
    output_dim=128,
    dropout=0.0
)
cnn_model = CNNNet(input_dim=1024, output_dim=128, dropout=0.3)
combined_model = CombinedNet(input_dim=256, hidden_dim=512, output_dim=1)  # 128 + 128 = 256

optimizer = torch.optim.Adam(
    list(chemprop_model.parameters()) +
    list(cnn_model.parameters()) +
    list(combined_model.parameters()),
    lr=0.0005,
    weight_decay=1e-4
)


best_mse = float('inf')
best_original_mse = float('inf')
best_model_file = 'best_model_chemprop.pth'
lambda_kl = 0.001

# Training loop
for epoch in range(1000):
    chemprop_model.train()
    cnn_model.train()
    combined_model.train()
    epoch_train_loss = 0.0
    num_batches = 0

    for batch_data, batch_ecfp in train_loader:
        optimizer.zero_grad()
        chemprop_output = chemprop_model(batch_data)  # [batch_size, 128]
        cnn_output = cnn_model(batch_ecfp)  # [batch_size, 128]
        combined_output = torch.cat((chemprop_output, cnn_output), dim=1)  # [batch_size, 256]
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
    chemprop_model.eval()
    cnn_model.eval()
    combined_model.eval()
    test_mse_total = 0.0

    # Collect all predictions and targets for metric calculation
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_data, batch_ecfp in test_loader:
            chemprop_output = chemprop_model(batch_data)
            cnn_output = cnn_model(batch_ecfp)
            combined_output = torch.cat((chemprop_output, cnn_output), dim=1)
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
            'chemprop_model_state_dict': chemprop_model.state_dict(),
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