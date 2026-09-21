import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


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


def evaluate_independent_test_set(model_path, test_csv_file, smiles_column='Smiles',
                                  target_column='pchembl', device='cpu',
                                  save_predictions=True, plot_results=True):
    """
    Evaluate performance on the independent test set

    Args:
        model_path: Path to the saved model
        test_csv_file: Path to the test set CSV file
        smiles_column: SMILES column name
        target_column: Target value column name
        device: Computing device ('cpu' or 'cuda')
        save_predictions: Whether to save prediction results
        plot_results: Whether to plot prediction results
    """

    print("=" * 80)
    print("Loading Model and Test Data...")
    print("=" * 80)

    # Load saved model - fix: add weights_only=False
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    scaler = checkpoint['scaler']

    # Initialize models
    chemprop_model = ChempropModel(
        atom_fdim=133,
        bond_fdim=6,
        hidden_size=300,
        depth=3,
        ffn_hidden_size=300,
        ffn_num_layers=2,
        output_dim=128,
        dropout=0.0
    ).to(device)

    cnn_model = CNNNet(input_dim=1024, output_dim=128, dropout=0.3).to(device)
    combined_model = CombinedNet(input_dim=256, hidden_dim=512, output_dim=1).to(device)

    # Load model parameters
    chemprop_model.load_state_dict(checkpoint['chemprop_model_state_dict'])
    cnn_model.load_state_dict(checkpoint['cnn_model_state_dict'])
    combined_model.load_state_dict(checkpoint['combined_model_state_dict'])

    # Set to evaluation mode
    chemprop_model.eval()
    cnn_model.eval()
    combined_model.eval()

    print(f"✓ Model loaded from: {model_path}")
    print(f"  - Trained MSE (normalized): {checkpoint.get('normalized_mse', 'N/A'):.4f}")
    print(f"  - Trained MSE (original): {checkpoint.get('original_mse', 'N/A'):.4f}")
    print(f"  - Trained MAE: {checkpoint.get('mae', 'N/A'):.4f}")
    print(f"  - Trained Pearson: {checkpoint.get('pearson', 'N/A'):.4f}")
    print(f"  - Trained R²: {checkpoint.get('r2', 'N/A'):.4f}")
    print()

    # Load test data
    test_df = pd.read_csv(test_csv_file)
    print(f"✓ Test data loaded from: {test_csv_file}")
    print(f"  - Number of samples: {len(test_df)}")
    print()

    # Prepare test data
    test_data_list = []
    failed_smiles = []

    for index, row in test_df.iterrows():
        smiles = str(row[smiles_column])
        try:
            atom_features_tensor, edge_index, edge_attr = smiles_to_graph(smiles)
            ecfp = get_ecfp(smiles)
            data = Data(x=atom_features_tensor, edge_index=edge_index, edge_attr=edge_attr)
            data.y_original = torch.tensor(row[target_column], dtype=torch.float)
            data.smiles = smiles
            test_data_list.append((data, ecfp))
        except Exception as e:
            failed_smiles.append((smiles, str(e)))

    if failed_smiles:
        print(f"⚠ Warning: {len(failed_smiles)} SMILES failed to process:")
        for smiles, error in failed_smiles[:5]:  # Only display the first 5
            print(f"  - {smiles}: {error}")
        if len(failed_smiles) > 5:
            print(f"  ... and {len(failed_smiles) - 5} more")
        print()

    test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)

    # Prediction
    print("=" * 80)
    print("Evaluating on Independent Test Set...")
    print("=" * 80)

    all_predictions = []
    all_targets = []
    all_smiles = []

    with torch.no_grad():
        for batch_data, batch_ecfp in test_loader:
            batch_data = batch_data.to(device)
            batch_ecfp = batch_ecfp.to(device)

            chemprop_output = chemprop_model(batch_data)
            cnn_output = cnn_model(batch_ecfp)
            combined_output = torch.cat((chemprop_output, cnn_output), dim=1)
            final_output = combined_model(combined_output)

            # Denormalize predicted values
            denormalized_preds = scaler.inverse_transform(final_output.cpu().numpy())
            original_targets = batch_data.y_original.view(-1, 1).cpu().numpy()

            all_predictions.extend(denormalized_preds.flatten())
            all_targets.extend(original_targets.flatten())
            all_smiles.extend(batch_data.smiles)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate evaluation metrics
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    pearson, p_value = pearsonr(all_targets, all_predictions)

    # Print results
    print("\n" + "=" * 80)
    print("INDEPENDENT TEST SET RESULTS")
    print("=" * 80)
    print(f"Number of samples:      {len(all_predictions)}")
    print(f"MSE:                    {mse:.4f}")
    print(f"RMSE:                   {rmse:.4f}")
    print(f"MAE:                    {mae:.4f}")
    print(f"Pearson Correlation:    {pearson:.4f} (p-value: {p_value:.4e})")
    print(f"R² Score:               {r2:.4f}")
    print("=" * 80)

    # Calculate residual statistics
    residuals = all_targets - all_predictions
    print("\nResidual Statistics:")
    print(f"Mean residual:          {np.mean(residuals):.4f}")
    print(f"Std residual:           {np.std(residuals):.4f}")
    print(f"Min residual:           {np.min(residuals):.4f}")
    print(f"Max residual:           {np.max(residuals):.4f}")
    print("=" * 80)

    # Save prediction results
    if save_predictions:
        results_df = pd.DataFrame({
            'SMILES': all_smiles,
            'True_Value': all_targets,
            'Predicted_Value': all_predictions,
            'Residual': residuals,
            'Absolute_Error': np.abs(residuals)
        })
        output_file = 'independent_test_predictions.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Predictions saved to: {output_file}")

    # Plot results
    if plot_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Predicted values vs. true values scatter plot
        ax1 = axes[0, 0]
        ax1.scatter(all_targets, all_predictions, alpha=0.5, s=20)
        min_val = min(all_targets.min(), all_predictions.min())
        max_val = max(all_targets.max(), all_predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('True Values', fontsize=12)
        ax1.set_ylabel('Predicted Values', fontsize=12)
        ax1.set_title(f'Predictions vs True Values\nR² = {r2:.4f}, Pearson = {pearson:.4f}', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Residual plot
        ax2 = axes[0, 1]
        ax2.scatter(all_predictions, residuals, alpha=0.5, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Values', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=12)
        ax2.set_title(f'Residual Plot\nMean = {np.mean(residuals):.4f}, Std = {np.std(residuals):.4f}', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # 3. Residual distribution histogram
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='r', linestyle='--', lw=2)
        ax3.set_xlabel('Residuals', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Residual Distribution', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # 4. Error distribution boxplot and metric summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        metrics_text = f"""
        PERFORMANCE METRICS
        {'=' * 40}

        MSE:        {mse:.4f}
        RMSE:       {rmse:.4f}
        MAE:        {mae:.4f}
        Pearson:    {pearson:.4f}
        R²:         {r2:.4f}

        RESIDUAL STATISTICS
        {'=' * 40}

        Mean:       {np.mean(residuals):.4f}
        Std:        {np.std(residuals):.4f}
        Min:        {np.min(residuals):.4f}
        Max:        {np.max(residuals):.4f}

        DATASET INFO
        {'=' * 40}

        Samples:    {len(all_predictions)}
        Failed:     {len(failed_smiles)}
        """
        ax4.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                 verticalalignment='center')

        plt.tight_layout()
        plot_file = 'independent_test_results.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Results plot saved to: {plot_file}")
        plt.show()

    # Return evaluation metrics
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'pearson': pearson,
        'r2': r2,
        'predictions': all_predictions,
        'targets': all_targets,
        'residuals': residuals
    }


if __name__ == "__main__":
    # Usage example
    model_path = 'best_model_chemprop.pth'  # Path to your model file
    test_csv_file = 'test_data.csv'  # Path to your independent test set

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Evaluate the independent test set
    results = evaluate_independent_test_set(
        model_path=model_path,
        test_csv_file=test_csv_file,
        smiles_column='Smiles',  # Adjust the column name according to your CSV file
        target_column='pchembl',  # Adjust the column name according to your CSV file
        device=device,
        save_predictions=True,
        plot_results=True
    )

    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)