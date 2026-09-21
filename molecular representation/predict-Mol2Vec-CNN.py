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
# Three-layer CNN model definition (must be exactly the same as during training)
# =====================================================
class ThreeLayerCNN(nn.Module):
    """Three-layer CNN for processing sequence embeddings; dropout=0.3"""

    def __init__(
            self,
            input_dim=300,
            num_filters_1=256,
            num_filters_2=128,
            num_filters_3=64,
            kernel_size=3,
            dropout=0.3
    ):
        super(ThreeLayerCNN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_filters_1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(num_filters_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=num_filters_1,
            out_channels=num_filters_2,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(num_filters_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(
            in_channels=num_filters_2,
            out_channels=num_filters_3,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn3 = nn.BatchNorm1d(num_filters_3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters_3, 1)

    def forward(self, x, lengths=None):
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        output = self.fc(x)
        return output.squeeze(-1)


# =====================================================
# Mol2Vec conversion function
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

    identifiers_alt = []
    for atom in dict_atoms:
        for r in radii:
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])
    return list(alternating_sentence)


def sentences2vec_sequence(sentences, model, unseen='UNK', max_len=100):
    """Convert a list of sentences into sequence vectors (for CNN)"""
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
# Extract sequence embeddings using Mol2Vec
# =====================================================
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
            if failed_count <= 3:
                print(f"\n   ⚠️  Conversion failed (index {idx}): {str(e)[:50]}")

    if failed_count > 0:
        print(f"   ⚠️  Conversion failed for {failed_count} molecules")

    if not sequences:
        print("   ❌ Error: No molecules were converted successfully!")
        sys.exit(1)

    print("   🔄 Converting to vector sequences...")
    padded_sequences, seq_lengths = sentences2vec_sequence(sequences, model, unseen='UNK', max_len=max_len)
    df_clean = df.iloc[valid_indices].reset_index(drop=True)

    print(f"   ✅ Successfully converted: {len(sequences)}/{len(smiles_list)} ({len(sequences) / len(smiles_list) * 100:.1f}%)")
    print(f"   📏 Sequence shape: {padded_sequences.shape}")
    print(f"   📏 Data type: {padded_sequences.dtype}")
    print(f"   📏 Memory usage: {padded_sequences.nbytes / (1024 ** 2):.2f} MB")

    return padded_sequences, seq_lengths, df_clean


# =====================================================
# Calculate evaluation metrics
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
# Independent test set evaluation
# =====================================================
def evaluate_test_set(
        model_path='best_model_cnn.pth',
        mol2vec_model_path='model_300dim.pkl',
        test_csv_path='test_data.csv',
        smiles_column='Smiles',
        target_column='pchembl',
        mol2vec_radius=1,
        max_sequence_length=100,
        output_csv='test_predictions.csv'
):
    """
    Load the trained model and evaluate it on the independent test set

    Parameters:
        model_path: Path to the trained model weights
        mol2vec_model_path: Path to the Mol2Vec model
        test_csv_path: Path to the test CSV file
        smiles_column: SMILES column name
        target_column: Target value column name
        mol2vec_radius: Mol2Vec radius parameter
        max_sequence_length: Maximum sequence length
        output_csv: Output CSV filename for prediction results
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("🧬 Independent Test Set Evaluation - Mol2Vec + Three-Layer CNN")
    print("=" * 70)
    print(f"🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # =====================================================
    # 1️⃣ Load Mol2Vec model
    # =====================================================
    print("\n📥 Loading Mol2Vec model...")
    try:
        mol2vec_model = word2vec.Word2Vec.load(mol2vec_model_path)
        embedding_dim = mol2vec_model.wv.vector_size
        print(f"✅ Mol2Vec model loaded successfully")
        print(f"   Embedding dimension: {embedding_dim}")
    except Exception as e:
        print(f"❌ Failed to load Mol2Vec model: {e}")
        sys.exit(1)

    # =====================================================
    # 2️⃣ Read test dataset
    # =====================================================
    print(f"\n📂 Reading test dataset: {test_csv_path}")
    try:
        test_df = pd.read_csv(test_csv_path, encoding='utf-8')
        print(f"✅ Test dataset: {len(test_df)} rows")

        # Check whether columns exist
        if smiles_column not in test_df.columns:
            print(f"❌ Column '{smiles_column}' does not exist")
            print(f"💡 Available columns: {', '.join(test_df.columns.tolist())}")
            sys.exit(1)
        if target_column not in test_df.columns:
            print(f"❌ Column '{target_column}' does not exist")
            print(f"💡 Available columns: {', '.join(test_df.columns.tolist())}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Failed to read test dataset: {e}")
        sys.exit(1)

    # =====================================================
    # 3️⃣ Extract test set features
    # =====================================================
    print("\n" + "=" * 70)
    print("🔄 Extracting Mol2Vec sequence embeddings from test set")
    print("=" * 70)

    test_sequences, test_lengths, test_df_clean = extract_mol2vec_sequences(
        test_df, smiles_column, mol2vec_model,
        radius=mol2vec_radius, max_len=max_sequence_length
    )
    test_labels = test_df_clean[target_column].values
    print(f"   ✅ Test sequences: {test_sequences.shape}")

    # =====================================================
    # 4️⃣ Load trained model
    # =====================================================
    print("\n" + "=" * 70)
    print("🏗️  Loading trained model")
    print("=" * 70)

    try:
        model = ThreeLayerCNN(
            input_dim=embedding_dim,
            num_filters_1=256,
            num_filters_2=128,
            num_filters_3=64,
            kernel_size=3,
            dropout=0.3
        ).to(device)

        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set to evaluation mode

        print(f"✅ Model loaded successfully: {model_path}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # =====================================================
    # 5️⃣ Prediction
    # =====================================================
    print("\n" + "=" * 70)
    print("🎯 Starting prediction")
    print("=" * 70)

    test_X = torch.FloatTensor(test_sequences).to(device)
    test_len = torch.LongTensor(test_lengths)

    with torch.no_grad():
        predictions = model(test_X, test_len).cpu().numpy()

    print(f"✅ Prediction completed: {len(predictions)} samples")

    # =====================================================
    # 6️⃣ Calculate evaluation metrics
    # =====================================================
    print("\n" + "=" * 70)
    print("📊 Test Set Performance Evaluation")
    print("=" * 70)

    metrics = calculate_metrics(predictions, test_labels)

    print(f"\n🎯 Test set results:")
    print(f"   MSE (Mean Squared Error):          {metrics['mse']:.6f}")
    print(f"   RMSE (Root Mean Squared Error):    {metrics['rmse']:.6f}")
    print(f"   MAE (Mean Absolute Error):         {metrics['mae']:.6f}")
    print(f"   Pearson correlation coefficient:  {metrics['pearson']:.6f}")
    print(f"   Pearson p-value:                   {metrics['p_value']:.6e}")
    print(f"   R² (Coefficient of Determination): {metrics['r2']:.6f}")

    # =====================================================
    # 7️⃣ Save prediction results
    # =====================================================
    results_df = test_df_clean.copy()
    results_df['Predicted'] = predictions
    results_df['Actual'] = test_labels
    results_df['Error'] = predictions - test_labels
    results_df['Abs_Error'] = np.abs(predictions - test_labels)
    results_df['Squared_Error'] = (predictions - test_labels) ** 2

    results_df.to_csv(output_csv, index=False)

    print(f"\n💾 Prediction results saved to: {output_csv}")

    # Statistical information
    print(f"\n📈 Prediction statistics:")
    print(f"   Predicted value range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"   Actual value range:    [{test_labels.min():.3f}, {test_labels.max():.3f}]")
    print(f"   Mean predicted value:  {predictions.mean():.3f}")
    print(f"   Mean actual value:     {test_labels.mean():.3f}")
    print(f"   Maximum error:         {np.max(np.abs(predictions - test_labels)):.3f}")
    print(f"   Minimum error:         {np.min(np.abs(predictions - test_labels)):.3f}")

    print("\n" + "=" * 70)
    print("🎉 Evaluation completed!")
    print("=" * 70)

    return metrics, predictions, test_labels


# =====================================================
# Main program
# =====================================================
if __name__ == "__main__":
    # =====================================================
    # Parameter configuration
    # =====================================================
    # Change the following paths to your actual paths
    MODEL_PATH = 'best_model_cnn.pth'  # Trained model weights
    MOL2VEC_MODEL_PATH = 'model_300dim.pkl'  # Mol2Vec model
    TEST_CSV_PATH = r'test_data.csv'  # Test dataset path

    SMILES_COLUMN = 'Smiles'  # SMILES column name
    TARGET_COLUMN = 'pchembl'  # Target value column name

    MOL2VEC_RADIUS = 1  # Must be consistent with training
    MAX_SEQUENCE_LENGTH = 100  # Must be consistent with training

    OUTPUT_CSV = 'test_predictions.csv'  # Output filename

    # =====================================================
    # Run evaluation
    # =====================================================
    metrics, predictions, actuals = evaluate_test_set(
        model_path=MODEL_PATH,
        mol2vec_model_path=MOL2VEC_MODEL_PATH,
        test_csv_path=TEST_CSV_PATH,
        smiles_column=SMILES_COLUMN,
        target_column=TARGET_COLUMN,
        mol2vec_radius=MOL2VEC_RADIUS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        output_csv=OUTPUT_CSV
    )