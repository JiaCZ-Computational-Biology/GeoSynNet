import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from torch_geometric.nn import global_max_pool
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
        results = one_of_k_encoding_unk(
            atom.GetSymbol(),
            ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Unknown']
        ) + \
        one_of_k_encoding_unk(
            atom.GetDegree(),
            [0, 1, 2, 3, 4, 5, 6]
        ) + \
        one_of_k_encoding_unk(
            atom.GetImplicitValence(),
            [0, 1, 2, 3, 4, 5, 6]
        ) + \
        one_of_k_encoding_unk(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ]
        ) + [atom.GetIsAromatic()] + \
        one_of_k_encoding_unk(
            atom.GetTotalNumHs(),
            [0, 1, 2, 3, 4]
        )

        atom_feats = np.array(results).astype(np.float32)
        atom_features.append(atom_feats)

        pos = conformer.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])

    atom_features = torch.tensor(
        atom_features,
        dtype=torch.float
    )

    coords = torch.tensor(
        coords,
        dtype=torch.float
    )

    adj_matrix = torch.zeros(
        (num_atoms, num_atoms),
        dtype=torch.float
    )

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj_matrix[start, end] = 1.0
        adj_matrix[end, start] = 1.0

    edge_index = adj_matrix.nonzero(
        as_tuple=False
    ).t().long()

    return atom_features, edge_index, coords


def get_ecfp(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Cannot generate molecule from SMILES: {smiles}"
        )

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=nBits
    )

    return torch.tensor(
        np.array(fp),
        dtype=torch.float
    ).view(1, -1)


def get_molecular_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Cannot generate molecule from SMILES: {smiles}"
        )

    mol_wt = Descriptors.MolWt(mol)
    mol_logp = Descriptors.MolLogP(mol)
    num_h_donors = Descriptors.NumHDonors(mol)
    num_h_acceptors = Descriptors.NumHAcceptors(mol)
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    mol_mr = Descriptors.MolMR(mol)

    try:
        fraction_csp3 = rdMolDescriptors.CalcFractionCsp3(mol)

    except:
        num_csp3 = sum(
            1 for atom in mol.GetAtoms()
            if atom.GetSymbol() == 'C'
            and atom.GetHybridization()
            == Chem.rdchem.HybridizationType.SP3
        )

        num_carbons = sum(
            1 for atom in mol.GetAtoms()
            if atom.GetSymbol() == 'C'
        )

        fraction_csp3 = (
            num_csp3 / num_carbons
            if num_carbons > 0
            else 0.0
        )

    descriptors = np.array(
        [
            mol_wt,
            mol_logp,
            num_h_donors,
            num_h_acceptors,
            num_rotatable_bonds,
            tpsa,
            mol_mr,
            fraction_csp3
        ],
        dtype=np.float32
    )

    return torch.tensor(
        descriptors,
        dtype=torch.float
    )


class EGNNLayer(nn.Module):

    def __init__(
        self,
        in_node_features,
        hidden_features,
        out_node_features
    ):
        super(EGNNLayer, self).__init__()

        self.in_node_features = in_node_features
        self.hidden_features = hidden_features
        self.out_node_features = out_node_features

        self.edge_mlp = nn.Sequential(
            nn.Linear(
                2 * in_node_features + 1,
                hidden_features
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_features,
                hidden_features
            ),
            nn.SiLU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(
                in_node_features + hidden_features,
                hidden_features
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_features,
                out_node_features
            )
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(
                hidden_features,
                hidden_features
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_features,
                1,
                bias=False
            )
        )

    def forward(
        self,
        h,
        coords,
        edge_index
    ):
        row, col = edge_index

        coord_diff = (
            coords[row]
            - coords[col]
        )

        radial = torch.sum(
            coord_diff ** 2,
            dim=1,
            keepdim=True
        )

        edge_feat = torch.cat(
            [
                h[row],
                h[col],
                radial
            ],
            dim=1
        )

        edge_msg = self.edge_mlp(
            edge_feat
        )

        coord_weights = self.coord_mlp(
            edge_msg
        )

        coord_diff_normalized = (
            coord_diff /
            (
                torch.sqrt(radial)
                + 1e-8
            )
        )

        coord_update = (
            coord_diff_normalized
            * coord_weights
        )

        coords_updated = coords.clone()

        coords_updated.index_add_(
            0,
            row,
            coord_update
        )

        agg_msg = torch.zeros(
            h.size(0),
            self.hidden_features,
            device=h.device
        )

        agg_msg.index_add_(
            0,
            row,
            edge_msg
        )

        node_feat = torch.cat(
            [
                h,
                agg_msg
            ],
            dim=1
        )

        h_updated = (
            self.node_mlp(node_feat)
            + h[:, :self.out_node_features]
        )

        return h_updated, coords_updated


class EGNN(nn.Module):

    def __init__(
        self,
        num_features_xd=35,
        hidden_dim=128,
        output_dim=128,
        num_layers=3,
        dropout=0.3
    ):
        super(EGNN, self).__init__()

        self.num_layers = num_layers

        self.dropout = nn.Dropout(
            dropout
        )

        self.node_embedding = nn.Linear(
            num_features_xd,
            hidden_dim
        )

        self.egnn_layers = nn.ModuleList()

        for i in range(num_layers):
            self.egnn_layers.append(
                EGNNLayer(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim
                )
            )

        self.fc_g1 = nn.Linear(
            hidden_dim,
            256
        )

        self.fc_g2 = nn.Linear(
            256,
            output_dim
        )

        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, coords, batch = (
            data.x,
            data.edge_index,
            data.coords,
            data.batch
        )

        h = self.relu(
            self.node_embedding(x)
        )

        for egnn_layer in self.egnn_layers:
            h, coords = egnn_layer(
                h,
                coords,
                edge_index
            )

            h = self.dropout(h)

        x = global_max_pool(
            h,
            batch
        )

        x = self.relu(
            self.fc_g1(x)
        )

        x = self.dropout(x)

        x = self.fc_g2(x)

        return x


class CNNNet(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim=128,
        dropout=0.3
    ):
        super(CNNNet, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding='same'
        )

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding='same'
        )

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding='same'
        )

        self.fc1 = nn.Linear(
            128 * input_dim,
            512
        )

        self.fc2 = nn.Linear(
            512,
            output_dim
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, ecfp):
        ecfp = (
            ecfp.squeeze(1)
            .unsqueeze(1)
        )

        x = self.relu(
            self.conv1(ecfp)
        )

        x = self.relu(
            self.conv2(x)
        )

        x = self.relu(
            self.conv3(x)
        )

        x = x.view(
            x.size(0),
            -1
        )

        x = self.dropout(
            self.relu(
                self.fc1(x)
            )
        )

        x = self.fc2(x)

        return x


class DescriptorMLP(nn.Module):

    def __init__(
        self,
        input_dim=8,
        hidden_dim=64,
        output_dim=32,
        dropout=0.3
    ):
        super(
            DescriptorMLP,
            self
        ).__init__()

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.fc3 = nn.Linear(
            hidden_dim,
            output_dim
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.bn1 = nn.BatchNorm1d(
            hidden_dim
        )

        self.bn2 = nn.BatchNorm1d(
            hidden_dim
        )

    def forward(self, x):
        x = self.relu(
            self.bn1(
                self.fc1(x)
            )
        )

        x = self.dropout(x)

        x = self.relu(
            self.bn2(
                self.fc2(x)
            )
        )

        x = self.dropout(x)

        x = self.fc3(x)

        return x


class CombinedNet(nn.Module):

    def __init__(
        self,
        input_dim=288,
        hidden_dim=512,
        output_dim=1
    ):
        super(
            CombinedNet,
            self
        ).__init__()

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            256
        )

        self.fc3 = nn.Linear(
            256,
            output_dim
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            0.3
        )

    def forward(self, x):
        x = self.dropout(
            self.relu(
                self.fc1(x)
            )
        )

        x = self.dropout(
            self.relu(
                self.fc2(x)
            )
        )

        x = self.fc3(x)

        return x


def evaluate_model(
    model_path,
    test_csv_file,
    smiles_column='Smiles',
    target_column='pchembl',
    device='cpu'
):
    print(
        f"Loading model from {model_path}..."
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    scaler = checkpoint[
        'scaler'
    ]

    descriptor_scaler = checkpoint[
        'descriptor_scaler'
    ]

    egnn_model = EGNN(
        num_features_xd=35,
        hidden_dim=128,
        output_dim=128,
        num_layers=3,
        dropout=0.3
    )

    cnn_model = CNNNet(
        input_dim=1024,
        output_dim=128,
        dropout=0.3
    )

    descriptor_mlp = DescriptorMLP(
        input_dim=8,
        hidden_dim=64,
        output_dim=32,
        dropout=0.3
    )

    combined_model = CombinedNet(
        input_dim=288,
        hidden_dim=512,
        output_dim=1
    )

    egnn_model.load_state_dict(
        checkpoint[
            'egnn_model_state_dict'
        ]
    )

    cnn_model.load_state_dict(
        checkpoint[
            'cnn_model_state_dict'
        ]
    )

    descriptor_mlp.load_state_dict(
        checkpoint[
            'descriptor_mlp_state_dict'
        ]
    )

    combined_model.load_state_dict(
        checkpoint[
            'combined_model_state_dict'
        ]
    )

    egnn_model.eval()
    cnn_model.eval()
    descriptor_mlp.eval()
    combined_model.eval()

    egnn_model.to(device)
    cnn_model.to(device)
    descriptor_mlp.to(device)
    combined_model.to(device)

    print(
        f"Loading test data from "
        f"{test_csv_file}..."
    )

    test_df = pd.read_csv(
        test_csv_file
    )

    test_data_list = []
    failed_count = 0

    for index, row in test_df.iterrows():

        smiles = str(
            row[smiles_column]
        )

        try:
            atom_features, edge_index, coords = (
                smiles_to_graph(
                    smiles
                )
            )

            ecfp = get_ecfp(
                smiles
            )

            descriptors = (
                get_molecular_descriptors(
                    smiles
                )
            )

            data = Data(
                x=atom_features,
                edge_index=edge_index,
                coords=coords
            )

            data.y_original = torch.tensor(
                row[target_column],
                dtype=torch.float
            )

            test_data_list.append(
                (
                    data,
                    ecfp,
                    descriptors
                )
            )

        except Exception as e:
            failed_count += 1

            print(
                f"Error processing SMILES "
                f"{smiles}: {e}"
            )

    print(
        f"Successfully processed "
        f"{len(test_data_list)} test samples"
    )

    if failed_count > 0:
        print(
            f"Failed to process "
            f"{failed_count} samples"
        )

    all_test_descriptors = torch.stack(
        [
            item[2]
            for item in test_data_list
        ]
    )

    all_test_descriptors_normalized = (
        descriptor_scaler.transform(
            all_test_descriptors.numpy()
        )
    )

    for i in range(
        len(test_data_list)
    ):
        data, ecfp, _ = (
            test_data_list[i]
        )

        normalized_desc = torch.tensor(
            all_test_descriptors_normalized[i],
            dtype=torch.float
        )

        test_data_list[i] = (
            data,
            ecfp,
            normalized_desc
        )

    test_loader = DataLoader(
        test_data_list,
        batch_size=64,
        shuffle=False
    )

    print(
        "\nEvaluating on independent "
        "test set..."
    )

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for (
            batch_data,
            batch_ecfp,
            batch_descriptors
        ) in test_loader:

            batch_data = (
                batch_data.to(device)
            )

            batch_ecfp = (
                batch_ecfp.to(device)
            )

            batch_descriptors = (
                batch_descriptors.to(device)
            )

            egnn_output = egnn_model(
                batch_data
            )

            cnn_output = cnn_model(
                batch_ecfp
            )

            descriptor_output = (
                descriptor_mlp(
                    batch_descriptors
                )
            )

            combined_output = torch.cat(
                (
                    egnn_output,
                    cnn_output,
                    descriptor_output
                ),
                dim=1
            )

            final_output = combined_model(
                combined_output
            )

            denormalized_preds = (
                scaler.inverse_transform(
                    final_output
                    .cpu()
                    .numpy()
                )
            )

            original_targets = (
                batch_data
                .y_original
                .view(-1, 1)
                .cpu()
                .numpy()
            )

            all_predictions.extend(
                denormalized_preds.flatten()
            )

            all_targets.extend(
                original_targets.flatten()
            )

    all_predictions = np.array(
        all_predictions
    )

    all_targets = np.array(
        all_targets
    )

    test_mse = mean_squared_error(
        all_targets,
        all_predictions
    )

    test_rmse = np.sqrt(
        test_mse
    )

    test_mae = mean_absolute_error(
        all_targets,
        all_predictions
    )

    test_r2 = r2_score(
        all_targets,
        all_predictions
    )

    test_pearson, p_value = pearsonr(
        all_targets,
        all_predictions
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "Independent Test Set Evaluation Results"
    )

    print(
        "=" * 60
    )

    print(
        f"Number of samples:      "
        f"{len(all_targets)}"
    )

    print(
        f"MSE:                    "
        f"{test_mse:.4f}"
    )

    print(
        f"RMSE:                   "
        f"{test_rmse:.4f}"
    )

    print(
        f"MAE:                    "
        f"{test_mae:.4f}"
    )

    print(
        f"R²:                     "
        f"{test_r2:.4f}"
    )

    print(
        f"Pearson Correlation:    "
        f"{test_pearson:.4f}"
    )

    print(
        f"P-value:                "
        f"{p_value:.4e}"
    )

    print(
        "=" * 60
    )

    results = {
        'mse': test_mse,
        'rmse': test_rmse,
        'mae': test_mae,
        'r2': test_r2,
        'pearson': test_pearson,
        'p_value': p_value,
        'predictions': all_predictions,
        'targets': all_targets,
        'num_samples': len(all_targets)
    }

    return results


def save_predictions(
    results,
    output_file='test_predictions.csv'
):
    df = pd.DataFrame(
        {
            'True_Value':
                results['targets'],

            'Predicted_Value':
                results['predictions'],

            'Absolute_Error':
                np.abs(
                    results['targets']
                    - results['predictions']
                )
        }
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nPredictions saved to "
        f"{output_file}"
    )


if __name__ == "__main__":

    model_path = (
        'best_model_egnn_with_descriptors.pth'
    )

    test_csv_file = (
        'test_data.csv'
    )

    smiles_column = 'Smiles'

    target_column = 'pchembl'

    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    print(
        f"Using device: {device}"
    )

    results = evaluate_model(
        model_path=model_path,
        test_csv_file=test_csv_file,
        smiles_column=smiles_column,
        target_column=target_column,
        device=device
    )

    save_predictions(
        results,
        output_file='independent_test_predictions.csv'
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    if 'original_mse' in checkpoint:

        print(
            "\n" + "=" * 60
        )

        print(
            "Training Set Best Performance "
            "(from checkpoint)"
        )

        print(
            "=" * 60
        )

        print(
            f"MSE:                    "
            f"{checkpoint['original_mse']:.4f}"
        )

        print(
            f"MAE:                    "
            f"{checkpoint['mae']:.4f}"
        )

        print(
            f"Pearson Correlation:    "
            f"{checkpoint['pearson']:.4f}"
        )

        print(
            f"R²:                     "
            f"{checkpoint['r2']:.4f}"
        )

        print(
            "=" * 60
        )