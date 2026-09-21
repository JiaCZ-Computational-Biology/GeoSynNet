import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import sys
import warnings
from scipy.stats import pearsonr
from rdkit import Chem
from rdkit.Chem import AllChem
from gensim.models import word2vec

warnings.filterwarnings('ignore')


# =====================================================
# 🧪 Mol2Vec conversion function
# =====================================================
def mol2alt_sentence_fixed(mol, radius=1):
    """Convert a molecule into a sentence (list of substructure identifiers)"""
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element

    identifiers_alt = []
    for atom in dict_atoms:
        for r in radii:
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])
    return list(alternating_sentence)


def sentences2vec_sequence(sentences, model, unseen='UNK', max_len=100):
    """Convert a list of sentences into sequence vectors"""
    if hasattr(model.wv, 'key_to_index'):
        keys = model.wv.key_to_index
        get_vector = lambda word: model.wv[word]
    else:
        keys = model.wv.vocab
        get_vector = lambda word: model.wv[word]

    sequences = []
    lengths = []

    for sentence in sentences:
        if unseen:
            sentence = [word for word in sentence if word in keys]

        if not sentence:
            sequences.append(np.zeros((1, model.wv.vector_size)))
            lengths.append(1)
        else:
            sentence = sentence[:max_len]
            word_vectors = [get_vector(word) for word in sentence]
            sequences.append(np.array(word_vectors))
            lengths.append(len(word_vectors))

    padded_sequences = []
    actual_max_len = min(max(lengths), max_len)

    for seq in sequences:
        if len(seq) < actual_max_len:
            padding = np.zeros((actual_max_len - len(seq), model.wv.vector_size))
            padded_seq = np.vstack([seq, padding])
        else:
            padded_seq = seq[:actual_max_len]
        padded_sequences.append(padded_seq)

    return np.array(padded_sequences), np.array(lengths)


def extract_mol2vec_sequences(df, smiles_column, model, radius=1, max_len=100):
    """Extract Mol2Vec sequence embeddings from a DataFrame"""
    original_count = len(df)
    df = df.dropna(subset=[smiles_column])
    df = df[df[smiles_column].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)

    removed_count = original_count - len(df)
    if removed_count > 0:
        print(f"   ⚠️  Removed {removed_count} invalid samples")

    smiles_list = df[smiles_column].astype(str).tolist()
    print(f"   ✅ Number of valid SMILES: {len(smiles_list)}")

    sequences = []
    valid_indices = []
    failed_count = 0

    print("   🔄 Starting molecule conversion...")
    for idx, smiles in enumerate(tqdm(smiles_list, desc="   Extracting Mol2Vec sequences")):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed_count += 1
                continue

            sentence = mol2alt_sentence_fixed(mol, radius=radius)
            sequences.append(sentence)
            valid_indices.append(idx)

        except Exception as e:
            failed_count += 1

    if failed_count > 0:
        print(f"   ⚠️  Conversion failed for {failed_count} molecules")

    if not sequences:
        print("   ❌ Error: No molecules were converted successfully!")
        sys.exit(1)

    padded_sequences, seq_lengths = sentences2vec_sequence(sequences, model, unseen='UNK', max_len=max_len)
    df_clean = df.iloc[valid_indices].reset_index(drop=True)

    print(f"   ✅ Successfully converted: {len(sequences)}/{len(smiles_list)} ({len(sequences) / len(smiles_list) * 100:.1f}%)")

    return padded_sequences, seq_lengths, df_clean


# =====================================================
# 🏗️ Three-layer BiLSTM model definition
# =====================================================
class ThreeLayerBiLSTM(nn.Module):
    """Three-layer BiLSTM for processing sequence embeddings"""

    def __init__(
            self,
            input_dim=300,
            hidden_dim_1=256,
            hidden_dim_2=128,
            hidden_dim_3=64,
            dropout=0.3,
            num_layers_per_block=1
    ):
        super(ThreeLayerBiLSTM, self).__init__()

        self.lstm1 = nn.LSTM(
            input_dim, hidden_dim_1, num_layers=num_layers_per_block,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers_per_block > 1 else 0
        )
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1 * 2)

        self.lstm2 = nn.LSTM(
            hidden_dim_1 * 2, hidden_dim_2, num_layers=num_layers_per_block,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers_per_block > 1 else 0
        )
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2 * 2)

        self.lstm3 = nn.LSTM(
            hidden_dim_2 * 2, hidden_dim_3, num_layers=num_layers_per_block,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers_per_block > 1 else 0
        )
        self.dropout3 = nn.Dropout(dropout)
        self.bn3 = nn.BatchNorm1d(hidden_dim_3 * 2)

        self.fc = nn.Linear(hidden_dim_3 * 2, 1)

    def forward(self, x, lengths=None):
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)

        if lengths is not None:
            batch_size = x.size(0)
            last_outputs = []
            for i in range(batch_size):
                last_outputs.append(lstm1_out[i, lengths[i] - 1, :])
            lstm1_last = torch.stack(last_outputs)
        else:
            lstm1_last = lstm1_out[:, -1, :]

        lstm1_last = self.bn1(lstm1_last)

        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)

        if lengths is not None:
            batch_size = x.size(0)
            last_outputs = []
            for i in range(batch_size):
                last_outputs.append(lstm2_out[i, lengths[i] - 1, :])
            lstm2_last = torch.stack(last_outputs)
        else:
            lstm2_last = lstm2_out[:, -1, :]

        lstm2_last = self.bn2(lstm2_last)

        lstm3_out, _ = self.lstm3(lstm2_out)
        lstm3_out = self.dropout3(lstm3_out)

        if lengths is not None:
            batch_size = x.size(0)
            last_outputs = []
            for i in range(batch_size):
                last_outputs.append(lstm3_out[i, lengths[i] - 1, :])
            lstm3_last = torch.stack(last_outputs)
        else:
            lstm3_last = lstm3_out[:, -1, :]

        lstm3_last = self.bn3(lstm3_last)

        output = self.fc(lstm3_last)
        return output.squeeze(-1)


# =====================================================
# 📈 Calculate evaluation metrics
# =====================================================
def calculate_metrics(predictions, actuals):
    """Calculate regression evaluation metrics"""
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    try:
        pearson_corr, _ = pearsonr(predictions, actuals)
    except:
        pearson_corr = 0.0

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr
    }


# =====================================================
# 🎬 Main program
# =====================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 BiLSTM Model Test - Independent Test Set Evaluation")
    print("=" * 70)

    # =====================================================
    # 1️⃣ Parameter settings
    # =====================================================
    test_csv_path = r"test_data.csv"
    model_path = "best_model_bilstm.pth"
    mol2vec_model_path = "model_300dim.pkl"

    smiles_column = "Smiles"
    target_column = "pchembl"

    mol2vec_radius = 1
    max_sequence_length = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")

    # =====================================================
    # 2️⃣ Load Mol2Vec model
    # =====================================================
    print("\n📥 Loading Mol2Vec model...")
    try:
        mol2vec_model = word2vec.Word2Vec.load(mol2vec_model_path)
        embedding_dim = mol2vec_model.wv.vector_size
        print(f"✅ Mol2Vec model loaded successfully (dimension: {embedding_dim})")
    except Exception as e:
        print(f"❌ Failed to load Mol2Vec model: {e}")
        sys.exit(1)

    # =====================================================
    # 3️⃣ Read test dataset
    # =====================================================
    print("\n📂 Reading test dataset...")
    try:
        test_df = pd.read_csv(test_csv_path, encoding='utf-8')
        print(f"✅ Test dataset: {len(test_df)} rows")
    except Exception as e:
        print(f"❌ Failed to read test dataset: {e}")
        sys.exit(1)

    if smiles_column not in test_df.columns or target_column not in test_df.columns:
        print(f"❌ Test dataset is missing required columns")
        sys.exit(1)

    # =====================================================
    # 4️⃣ Extract test set sequence embeddings
    # =====================================================
    print("\n🔄 Extracting Mol2Vec sequence embeddings from test set...")
    test_sequences, test_lengths, test_df_clean = extract_mol2vec_sequences(
        test_df, smiles_column, mol2vec_model,
        radius=mol2vec_radius, max_len=max_sequence_length
    )
    test_labels = test_df_clean[target_column].values
    print(f"   ✅ Test sequences: {test_sequences.shape}, samples: {len(test_labels)}")

    # =====================================================
    # 5️⃣ Load trained model
    # =====================================================
    print("\n🏗️  Loading trained BiLSTM model...")
    try:
        model = ThreeLayerBiLSTM(
            input_dim=embedding_dim,
            hidden_dim_1=256,
            hidden_dim_2=128,
            hidden_dim_3=64,
            dropout=0.3
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # =====================================================
    # 6️⃣ Model prediction
    # =====================================================
    print("\n🔮 Starting prediction...")
    with torch.no_grad():
        test_X = torch.FloatTensor(test_sequences).to(device)
        test_len = torch.LongTensor(test_lengths)
        predictions = model(test_X, test_len).cpu().numpy()
        actual = test_labels

    print(f"✅ Prediction completed (samples: {len(predictions)})")

    # =====================================================
    # 7️⃣ Calculate evaluation metrics
    # =====================================================
    print("\n" + "=" * 70)
    print("📊 Test Set Evaluation Metrics")
    print("=" * 70)

    metrics = calculate_metrics(predictions, actual)

    print(f"\n🎯 Test set performance:")
    print(f"   MSE (Mean Squared Error):          {metrics['mse']:.6f}")
    print(f"   MAE (Mean Absolute Error):        {metrics['mae']:.6f}")
    print(f"   Pearson correlation coefficient: {metrics['pearson']:.6f}")
    print(f"   R² (Coefficient of Determination): {metrics['r2']:.6f}")

    # =====================================================
    # 8️⃣ Save prediction results
    # =====================================================
    print("\n💾 Saving prediction results...")
    results_df = test_df_clean.copy()
    results_df['Predicted'] = predictions
    results_df['Actual'] = actual
    results_df['Error'] = predictions - actual
    results_df['Abs_Error'] = np.abs(predictions - actual)

    output_csv = 'test_predictions.csv'
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ Prediction results saved to: {output_csv}")

    # Save evaluation metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_csv = 'test_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"✅ Evaluation metrics saved to: {metrics_csv}")

    print("\n" + "=" * 70)
    print("🎉 Test completed!")
    print("=" * 70)