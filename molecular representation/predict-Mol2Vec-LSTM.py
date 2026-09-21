import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from gensim.models import word2vec
import warnings

warnings.filterwarnings('ignore')


# =====================================================
# 🏗️ Three-layer LSTM model definition (exactly the same as the training code)
# =====================================================
class ThreeLayerLSTM(nn.Module):
    """Three-layer LSTM for processing sequence embeddings; dropout=0.3"""

    def __init__(
            self,
            input_dim=300,
            hidden_dim_1=256,
            hidden_dim_2=128,
            hidden_dim_3=64,
            dropout=0.3,
            num_layers_per_block=1
    ):
        super(ThreeLayerLSTM, self).__init__()

        # First LSTM layer (unidirectional)
        self.lstm1 = nn.LSTM(
            input_dim,
            hidden_dim_1,
            num_layers=num_layers_per_block,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers_per_block > 1 else 0
        )
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)

        # Second LSTM layer (unidirectional)
        self.lstm2 = nn.LSTM(
            hidden_dim_1,
            hidden_dim_2,
            num_layers=num_layers_per_block,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers_per_block > 1 else 0
        )
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)

        # Third LSTM layer (unidirectional)
        self.lstm3 = nn.LSTM(
            hidden_dim_2,
            hidden_dim_3,
            num_layers=num_layers_per_block,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers_per_block > 1 else 0
        )
        self.dropout3 = nn.Dropout(dropout)
        self.bn3 = nn.BatchNorm1d(hidden_dim_3)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim_3, 1)

    def forward(self, x, lengths=None):
        # First LSTM layer
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

        # Second LSTM layer
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

        # Third LSTM layer
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

        # Output layer
        output = self.fc(lstm3_last)
        return output.squeeze(-1)


# =====================================================
# 🧪 Mol2Vec conversion function
# =====================================================
def mol2alt_sentence_fixed(mol, radius=1):
    """Convert a molecule into a "sentence" (list of substructure identifiers)"""
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element

    # Generate identifiers
    identifiers_alt = []
    for atom in dict_atoms:
        for r in radii:
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])
    return list(alternating_sentence)


def sentences2vec_sequence(sentences, model, unseen='UNK', max_len=100):
    """Convert a list of sentences into sequence vectors (for LSTM)"""
    # Compatible with different versions
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
            sequences.append(np.zeros((1, model.wv.vector_size), dtype=np.float32))
            lengths.append(1)
        else:
            sentence = sentence[:max_len]
            word_vectors = [get_vector(word) for word in sentence]
            sequences.append(np.array(word_vectors, dtype=np.float32))
            lengths.append(len(word_vectors))

    # Pad to a uniform length
    padded_sequences = []
    actual_max_len = min(max(lengths), max_len)

    for seq in sequences:
        if len(seq) < actual_max_len:
            padding = np.zeros((actual_max_len - len(seq), model.wv.vector_size), dtype=np.float32)
            padded_seq = np.vstack([seq, padding])
        else:
            padded_seq = seq[:actual_max_len]
        padded_sequences.append(padded_seq)

    return np.array(padded_sequences, dtype=np.float32), np.array(lengths, dtype=np.int32)


# =====================================================
# 📊 Extract sequence embeddings
# =====================================================
def extract_mol2vec_sequences(df, smiles_column, model, radius=1, max_len=100):
    """Extract Mol2Vec sequence embeddings from a DataFrame"""

    # Data cleaning
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
    for idx, smiles in enumerate(smiles_list):
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
            if failed_count <= 3:
                print(f"\n   ⚠️  Conversion failed (index {idx}): {str(e)[:50]}")

    if failed_count > 0:
        print(f"   ⚠️  Conversion failed for {failed_count} molecules")

    if not sequences:
        print("   ❌ Error: No molecules were converted successfully!")
        sys.exit(1)

    # Convert to padded sequences
    print("   🔄 Converting to vector sequences...")
    padded_sequences, seq_lengths = sentences2vec_sequence(sequences, model, unseen='UNK', max_len=max_len)
    df_clean = df.iloc[valid_indices].reset_index(drop=True)

    print(f"   ✅ Successfully converted: {len(sequences)}/{len(smiles_list)} ({len(sequences) / len(smiles_list) * 100:.1f}%)")
    print(f"   📏 Sequence shape: {padded_sequences.shape}")

    return padded_sequences, seq_lengths, df_clean


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
        pearson_corr, p_value = pearsonr(predictions, actuals)
    except:
        pearson_corr = 0.0
        p_value = 1.0

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr,
        'p_value': p_value
    }


# =====================================================
# 🎬 Main program - independent test set evaluation
# =====================================================
if __name__ == "__main__":

    # =====================================================
    # 1️⃣ Parameter settings
    # =====================================================
    # Modify the following paths
    test_csv_path = r"D:\pycharm\gutingle\pythonProject2\新\test_data.csv"  # Independent test set path
    model_path = "best_model_lstm.pth"  # Trained model path
    mol2vec_model_path = "model_300dim.pkl"  # Mol2Vec model path

    smiles_column = "Smiles"  # SMILES column name
    target_column = "pchembl"  # Target value column name

    # Model parameters (must be consistent with training)
    embedding_dim = 300
    hidden_dim_1 = 256
    hidden_dim_2 = 128
    hidden_dim_3 = 64
    dropout = 0.3

    mol2vec_radius = 1
    max_sequence_length = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("🧪 Independent Test Set Evaluation - Mol2Vec + Three-Layer LSTM")
    print("=" * 70)
    print(f"🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # =====================================================
    # 2️⃣ Read independent test set
    # =====================================================
    print("\n📂 Reading independent test set...")

    try:
        test_df = pd.read_csv(test_csv_path, encoding='utf-8')
        print(f"✅ Test set: {len(test_df)} rows")
    except Exception as e:
        print(f"❌ Failed to read test set: {e}")
        sys.exit(1)

    # Check whether columns exist
    if smiles_column not in test_df.columns:
        print(f"❌ Column '{smiles_column}' does not exist in the test set")
        print(f"💡 Available columns: {', '.join(test_df.columns.tolist())}")
        sys.exit(1)
    if target_column not in test_df.columns:
        print(f"❌ Column '{target_column}' does not exist in the test set")
        print(f"💡 Available columns: {', '.join(test_df.columns.tolist())}")
        sys.exit(1)

    # =====================================================
    # 3️⃣ Load Mol2Vec model
    # =====================================================
    print("\n" + "=" * 70)
    print("📥 Loading Mol2Vec model")
    print("=" * 70)

    try:
        mol2vec_model = word2vec.Word2Vec.load(mol2vec_model_path)
        print(f"✅ Mol2Vec model loaded successfully!")
        print(f"   Vector dimension: {mol2vec_model.wv.vector_size}")
    except Exception as e:
        print(f"❌ Failed to load Mol2Vec model: {e}")
        sys.exit(1)

    # =====================================================
    # 4️⃣ Extract test set sequence embeddings
    # =====================================================
    print("\n" + "=" * 70)
    print("🔄 Extracting Mol2Vec sequence embeddings from test set")
    print("=" * 70)

    test_sequences, test_lengths, test_df_clean = extract_mol2vec_sequences(
        test_df, smiles_column, mol2vec_model,
        radius=mol2vec_radius, max_len=max_sequence_length
    )
    test_labels = test_df_clean[target_column].values
    print(f"   ✅ Test set sequences: {test_sequences.shape}")

    # =====================================================
    # 5️⃣ Load trained model
    # =====================================================
    print("\n" + "=" * 70)
    print("🔄 Loading trained model")
    print("=" * 70)

    try:
        model = ThreeLayerLSTM(
            input_dim=embedding_dim,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            hidden_dim_3=hidden_dim_3,
            dropout=dropout
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loaded successfully!")
        print(f"   Model parameters: {total_params:,}")
        print(f"   Model path: {model_path}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # =====================================================
    # 6️⃣ Predict on independent test set
    # =====================================================
    print("\n" + "=" * 70)
    print("🔮 Starting prediction...")
    print("=" * 70)

    with torch.no_grad():
        test_X = torch.FloatTensor(test_sequences).to(device)
        test_len = torch.LongTensor(test_lengths)

        predictions = model(test_X, test_len).cpu().numpy()
        actual = test_labels

    print(f"✅ Prediction completed!")
    print(f"   Number of predicted samples: {len(predictions)}")

    # =====================================================
    # 7️⃣ Calculate evaluation metrics
    # =====================================================
    print("\n" + "=" * 70)
    print("📊 Independent Test Set Performance Evaluation")
    print("=" * 70)

    metrics = calculate_metrics(predictions, actual)

    print(f"\n🎯 Evaluation metrics:")
    print(f"{'Metric':<15} {'Value':<15}")
    print("-" * 30)
    print(f"{'MSE':<15} {metrics['mse']:<15.6f}")
    print(f"{'RMSE':<15} {metrics['rmse']:<15.6f}")
    print(f"{'MAE':<15} {metrics['mae']:<15.6f}")
    print(f"{'R²':<15} {metrics['r2']:<15.6f}")
    print(f"{'Pearson':<15} {metrics['pearson']:<15.6f}")
    print(f"{'P-value':<15} {metrics['p_value']:<15.6e}")
    print("-" * 30)

    # =====================================================
    # 8️⃣ Save prediction results
    # =====================================================
    print("\n💾 Saving prediction results...")

    results_df = test_df_clean.copy()
    results_df['Predicted'] = predictions
    results_df['Actual'] = actual
    results_df['Error'] = predictions - actual
    results_df['Abs_Error'] = np.abs(predictions - actual)
    results_df['Squared_Error'] = (predictions - actual) ** 2

    output_path = 'test_predictions_lstm.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Prediction results saved to: {output_path}")

    # =====================================================
    # 9️⃣ Additional statistical information
    # =====================================================
    print("\n" + "=" * 70)
    print("📈 Additional Statistical Information")
    print("=" * 70)

    print(f"\nActual value statistics:")
    print(f"   Mean:               {np.mean(actual):.4f}")
    print(f"   Standard deviation: {np.std(actual):.4f}")
    print(f"   Minimum:            {np.min(actual):.4f}")
    print(f"   Maximum:            {np.max(actual):.4f}")

    print(f"\nPredicted value statistics:")
    print(f"   Mean:               {np.mean(predictions):.4f}")
    print(f"   Standard deviation: {np.std(predictions):.4f}")
    print(f"   Minimum:            {np.min(predictions):.4f}")
    print(f"   Maximum:            {np.max(predictions):.4f}")

    print(f"\nError statistics:")
    errors = predictions - actual
    print(f"   Mean error:              {np.mean(errors):.4f}")
    print(f"   Error standard deviation:{np.std(errors):.4f}")
    print(f"   Maximum positive error:  {np.max(errors):.4f}")
    print(f"   Maximum negative error:  {np.min(errors):.4f}")

    # Calculate the proportion of samples within different error ranges
    abs_errors = np.abs(errors)
    thresholds = [0.5, 1.0, 1.5, 2.0]
    print(f"\nError distribution:")
    for threshold in thresholds:
        ratio = np.sum(abs_errors <= threshold) / len(abs_errors) * 100
        print(f"   |Error| ≤ {threshold}: {ratio:.2f}%")

    print("\n" + "=" * 70)
    print("🎉 Independent test set evaluation completed!")
    print("=" * 70)

    # =====================================================
    # 🔟 Generate evaluation report
    # =====================================================
    print("\n💾 Generating evaluation report...")

    report_path = 'test_evaluation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Independent Test Set Evaluation Report - Mol2Vec + Three-Layer LSTM\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Test set path: {test_csv_path}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Number of test samples: {len(predictions)}\n\n")

        f.write("=" * 70 + "\n")
        f.write("Evaluation Metrics\n")
        f.write("=" * 70 + "\n")
        f.write(f"MSE:        {metrics['mse']:.6f}\n")
        f.write(f"RMSE:       {metrics['rmse']:.6f}\n")
        f.write(f"MAE:        {metrics['mae']:.6f}\n")
        f.write(f"R²:         {metrics['r2']:.6f}\n")
        f.write(f"Pearson:    {metrics['pearson']:.6f}\n")
        f.write(f"P-value:    {metrics['p_value']:.6e}\n\n")

        f.write("=" * 70 + "\n")
        f.write("Statistical Information\n")
        f.write("=" * 70 + "\n")
        f.write(f"\nActual values:\n")
        f.write(f"  Mean:               {np.mean(actual):.4f}\n")
        f.write(f"  Standard deviation: {np.std(actual):.4f}\n")
        f.write(f"  Range:              [{np.min(actual):.4f}, {np.max(actual):.4f}]\n")

        f.write(f"\nPredicted values:\n")
        f.write(f"  Mean:               {np.mean(predictions):.4f}\n")
        f.write(f"  Standard deviation: {np.std(predictions):.4f}\n")
        f.write(f"  Range:              [{np.min(predictions):.4f}, {np.max(predictions):.4f}]\n")

        f.write(f"\nError analysis:\n")
        f.write(f"  Mean error:               {np.mean(errors):.4f}\n")
        f.write(f"  Error standard deviation: {np.std(errors):.4f}\n")
        f.write(f"  Error range:              [{np.min(errors):.4f}, {np.max(errors):.4f}]\n")

        f.write(f"\nError distribution:\n")
        for threshold in thresholds:
            ratio = np.sum(abs_errors <= threshold) / len(abs_errors) * 100
            f.write(f"  |Error| ≤ {threshold}: {ratio:.2f}%\n")

    print(f"✅ Evaluation report saved to: {report_path}")

    print("\n" + "=" * 70)
    print("📄 Generated files:")
    print(f"   1. {output_path} - Detailed prediction results")
    print(f"   2. {report_path} - Evaluation report")
    print("=" * 70)