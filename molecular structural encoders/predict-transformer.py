import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import global_max_pool, TransformerConv
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr
import pickle


def one_of_k_encoding_unk(x, valid_entries):
    if x not in valid_entries:
        x = 'Unknown'
    return [1 if entry == x else 0 for entry in valid_entries]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    num_atoms = mol.GetNumAtoms()
    atom_features = []

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

    atom_features = torch.tensor(atom_features, dtype=torch.float)
    adj_matrix = torch.zeros((num_atoms, num_atoms), dtype=torch.float)

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj_matrix[start, end] = 1.0
        adj_matrix[end, start] = 1.0

    edge_index = adj_matrix.nonzero(as_tuple=False).t().long()
    return atom_features, edge_index


def get_ecfp(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot generate molecule from SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return torch.tensor(np.array(fp), dtype=torch.float).view(1, -1)


class GraphTransformer(nn.Module):
    """Graph Transformer for molecular property prediction"""

    def __init__(self, num_features_xd=35, hidden_dim=128, output_dim=128, num_layers=3,
                 heads=4, dropout=0.3, edge_dim=None):
        super(GraphTransformer, self).__init__()

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Initial node embedding
        self.node_embedding = nn.Linear(num_features_xd, hidden_dim)

        # Graph Transformer layers
        self.transformer_layers = nn.ModuleList()
        for i in range(num_layers):
            self.transformer_layers.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    beta=True
                )
            )

        # Layer normalization for each transformer layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Output layers
        self.fc_g1 = nn.Linear(hidden_dim, 256)
        self.fc_g2 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initial embedding
        h = self.relu(self.node_embedding(x))

        # Graph Transformer layers with residual connections
        for i, transformer_layer in enumerate(self.transformer_layers):
            h_out = transformer_layer(h, edge_index)
            h_out = self.dropout(h_out)
            h = h + h_out
            h = self.layer_norms[i](h)

        # Global pooling
        x = global_max_pool(h, batch)

        # Output to 128 dimensions
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        return x


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
    Load the saved model and evaluate it on an independent test set

    Args:
        model_path: Path to the saved model file
        test_csv_file: Path to the test set CSV file
        smiles_column: SMILES column name
        target_column: Target value column name
        batch_size: Batch size
        device: Computing device ('cpu' or 'cuda')

    Returns:
        dict: Dictionary containing all evaluation metrics
    """

    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load saved model - add weights_only=False
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Initialize models
    graph_transformer_model = GraphTransformer(
        num_features_xd=35,
        hidden_dim=128,
        output_dim=128,
        num_layers=3,
        heads=4,
        dropout=0.3
    ).to(device)

    cnn_model = CNNNet(input_dim=1024, output_dim=128, dropout=0.3).to(device)
    combined_model = CombinedNet(input_dim=256, hidden_dim=512, output_dim=1).to(device)

    # Load model parameters
    graph_transformer_model.load_state_dict(checkpoint['graph_transformer_model_state_dict'])
    cnn_model.load_state_dict(checkpoint['cnn_model_state_dict'])
    combined_model.load_state_dict(checkpoint['combined_model_state_dict'])

    # Load scaler
    scaler = checkpoint['scaler']

    # Set models to evaluation mode
    graph_transformer_model.eval()
    cnn_model.eval()
    combined_model.eval()

    print("Models loaded successfully!")
    print(f"\nModel performance on validation set:")
    print(f"  MSE (Original): {checkpoint['original_mse']:.4f}")
    print(f"  MAE: {checkpoint['mae']:.4f}")
    print(f"  Pearson: {checkpoint['pearson']:.4f}")
    print(f"  R²: {checkpoint['r2']:.4f}")

    # Read test data
    print(f"\nLoading test data from {test_csv_file}...")
    test_df = pd.read_csv(test_csv_file)
    print(f"Test set size: {len(test_df)} samples")

    # Prepare test data
    test_data_list = []
    failed_count = 0

    for index, row in test_df.iterrows():
        smiles = str(row[smiles_column])
        try:
            atom_features, edge_index = smiles_to_graph(smiles)
            ecfp = get_ecfp(smiles)
            data = Data(x=atom_features, edge_index=edge_index)
            data.y_original = torch.tensor(row[target_column], dtype=torch.float)
            test_data_list.append((data, ecfp))
        except Exception as e:
            failed_count += 1
            print(f"Error processing SMILES {smiles}: {e}")

    print(f"Successfully processed: {len(test_data_list)} samples")
    if failed_count > 0:
        print(f"Failed to process: {failed_count} samples")

    # Create data loader
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)

    # Make predictions
    print("\nMaking predictions on test set...")
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_data, batch_ecfp in test_loader:
            batch_data = batch_data.to(device)
            batch_ecfp = batch_ecfp.to(device)

            # Forward propagation
            graph_output = graph_transformer_model(batch_data)
            cnn_output = cnn_model(batch_ecfp)
            combined_output = torch.cat((graph_output, cnn_output), dim=1)
            final_output = combined_model(combined_output)

            # Denormalize predicted values
            denormalized_preds = scaler.inverse_transform(final_output.cpu().numpy())
            original_targets = batch_data.y_original.view(-1, 1).cpu().numpy()

            all_predictions.extend(denormalized_preds.flatten())
            all_targets.extend(original_targets.flatten())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate evaluation metrics
    print("\nCalculating evaluation metrics...")
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    pearson, p_value = pearsonr(all_targets, all_predictions)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS ON INDEPENDENT TEST SET")
    print("=" * 60)
    print(f"Number of samples: {len(all_targets)}")
    print(f"\nMSE (Mean Squared Error):       {mse:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error):      {mae:.4f}")
    print(f"Pearson Correlation:            {pearson:.4f} (p-value: {p_value:.2e})")
    print(f"R² Score:                       {r2:.4f}")
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
    print(f"\nPredictions saved to {output_file}")

    # Return evaluation metrics dictionary
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'pearson': pearson,
        'pearson_pvalue': p_value,
        'r2': r2,
        'n_samples': len(all_targets),
        'predictions': all_predictions,
        'targets': all_targets
    }

    return metrics


if __name__ == "__main__":
    # Set file paths
    model_path = 'best_model_graph_transformer.pth'
    test_csv_file = 'test_data.csv'  # Independent test set file

    # Evaluate model
    metrics = evaluate_model(
        model_path=model_path,
        test_csv_file=test_csv_file,
        smiles_column='Smiles',
        target_column='pchembl',
        batch_size=64,
        device='cpu'  # Use 'cuda' if a GPU is available; otherwise use 'cpu'
    )

    # Further analyze results
    print("\nAdditional Statistics:")
    print(f"Mean Prediction:  {metrics['predictions'].mean():.4f}")
    print(f"Mean True Value:  {metrics['targets'].mean():.4f}")
    print(f"Std Prediction:   {metrics['predictions'].std():.4f}")
    print(f"Std True Value:   {metrics['targets'].std():.4f}")
    print(f"Min Error:        {np.min(np.abs(metrics['targets'] - metrics['predictions'])):.4f}")
    print(f"Max Error:        {np.max(np.abs(metrics['targets'] - metrics['predictions'])):.4f}")