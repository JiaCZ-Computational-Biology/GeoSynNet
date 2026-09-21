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
# 🏗️ Model definition (must be consistent with training)
# =====================================================
class EmbeddingMLP(nn.Module):
    """Two-layer MLP for processing embedding vectors; dropout=0.3"""

    def __init__(
            self,
            embedding_dim=300,
            hidden_dim_1=512,
            hidden_dim_2=256,
            dropout=0.3
    ):
        super(EmbeddingMLP, self).__init__()

        self.fc1 = nn.Linear(embedding_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden_dim_2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc_out(x)
        return x.squeeze(-1)


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

    identifiers_alt = []
    for atom in dict_atoms:
        for r in radii:
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])
    return list(alternating_sentence)


def sentences2vec_fixed(sentences, model, unseen='UNK'):
    """Convert a list of sentences into vectors"""
    if hasattr(model.wv, 'key_to_index'):
        keys = model.wv.key_to_index
        get_vector = lambda word: model.wv[word]
    else:
        keys = model.wv.vocab
        get_vector = lambda word: model.wv[word]

    vectors = []
    for sentence in sentences:
        if unseen:
            sentence = [word for word in sentence if word in keys]

        if not sentence:
            vectors.append(np.zeros(model.wv.vector_size))
        else:
            word_vectors = [get_vector(word) for word in sentence]
            vectors.append(np.mean(word_vectors, axis=0))

    return np.array(vectors)


# =====================================================
# 📊 Extract test set embedding vectors
# =====================================================
def extract_test_embeddings(df, smiles_column, model, radius=1):
    """Extract Mol2Vec embedding vectors from the test set DataFrame"""

    original_count = len(df)
    df = df.dropna(subset=[smiles_column])
    df = df[df[smiles_column].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)

    removed_count = original_count - len(df)
    if removed_count > 0:
        print(f"   ⚠️  Removed {removed_count} invalid samples")

    smiles_list = df[smiles_column].astype(str).tolist()
    print(f"   ✅ Number of valid SMILES: {len(smiles_list)}")

    vectors = []
    valid_indices = []
    failed_count = 0

    print("   🔄 Starting molecule conversion...")
    for idx, smiles in enumerate(tqdm(smiles_list, desc="   Extracting Mol2Vec embeddings")):
        try:
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                failed_count += 1
                continue

            sentence = mol2alt_sentence_fixed(mol, radius=radius)
            vec = sentences2vec_fixed([sentence], model, unseen='UNK')[0]

            vectors.append(vec)
            valid_indices.append(idx)

        except Exception as e:
            failed_count += 1
            if failed_count <= 3:
                print(f"\n   ⚠️  Conversion failed (index {idx}): {str(e)[:50]}")

    if failed_count > 0:
        print(f"   ⚠️  Conversion failed for {failed_count} molecules")

    if not vectors:
        print("   ❌ Error: No molecules were converted successfully!")
        sys.exit(1)

    vectors_array = np.array(vectors)
    df_clean = df.iloc[valid_indices].reset_index(drop=True)

    print(f"   ✅ Successfully converted: {len(vectors)}/{len(smiles_list)} ({len(vectors) / len(smiles_list) * 100:.1f}%)")

    return vectors_array, df_clean


# =====================================================
# 📈 Calculate evaluation metrics
# =====================================================
def calculate_metrics(predictions, actuals):
    """Calculate regression evaluation metrics"""
    # MSE
    mse = np.mean((predictions - actuals) ** 2)

    # MAE
    mae = np.mean(np.abs(predictions - actuals))

    # R² score
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Pearson correlation coefficient
    try:
        pearson_corr, p_value = pearsonr(predictions, actuals)
    except:
        pearson_corr = 0.0
        p_value = 1.0

    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr,
        'pearson_pvalue': p_value
    }


# =====================================================
# 🎯 Main test function
# =====================================================
def test_model(model_path, test_csv_path, mol2vec_model_path,
               smiles_column='Smiles', target_column='pchembl',
               mol2vec_radius=1, device='cuda'):
    """
    Evaluate the model on an independent test set

    Parameters:
        model_path: Path to the trained model (.pth file)
        test_csv_path: Path to the test CSV file
        mol2vec_model_path: Path to the pretrained Mol2Vec model
        smiles_column: SMILES column name
        target_column: Target value column name
        mol2vec_radius: Mol2Vec radius parameter
        device: 'cuda' or 'cpu'
    """

    print("=" * 70)
    print("🧪 Independent Test Set Evaluation")
    print("=" * 70)
    print(f"🔧 Device: {device}")
    print(f"📦 PyTorch version: {torch.__version__}")
    print("=" * 70)

    # =====================================================
    # 1️⃣ Load Mol2Vec model
    # =====================================================
    print("\n📥 Loading Mol2Vec model...")
    try:
        mol2vec_model = word2vec.Word2Vec.load(mol2vec_model_path)
        embedding_dim = mol2vec_model.wv.vector_size
        print(f"✅ Mol2Vec model loaded successfully!")
        print(f"   Vector dimension: {embedding_dim}")
    except Exception as e:
        print(f"❌ Failed to load Mol2Vec model: {e}")
        sys.exit(1)

    # =====================================================
    # 2️⃣ Read test set
    # =====================================================
    print("\n📂 Reading test set...")
    try:
        test_df = pd.read_csv(test_csv_path, encoding='utf-8')
        print(f"✅ Test set: {len(test_df)} rows")
        print(f"📊 Column names: {test_df.columns.tolist()}")
    except Exception as e:
        print(f"❌ Failed to read test set: {e}")
        sys.exit(1)

    # Check whether columns exist
    if smiles_column not in test_df.columns:
        print(f"❌ Column '{smiles_column}' does not exist")
        print(f"💡 Available columns: {', '.join(test_df.columns.tolist())}")
        sys.exit(1)
    if target_column not in test_df.columns:
        print(f"❌ Column '{target_column}' does not exist")
        print(f"💡 Available columns: {', '.join(test_df.columns.tolist())}")
        sys.exit(1)

    # =====================================================
    # 3️⃣ Extract test set embedding vectors
    # =====================================================
    print("\n" + "=" * 70)
    print("🔄 Extracting test set embedding vectors")
    print("=" * 70)

    test_embeddings, test_df_clean = extract_test_embeddings(
        test_df, smiles_column, mol2vec_model, radius=mol2vec_radius
    )
    test_labels = test_df_clean[target_column].values

    print(f"   ✅ Test set embeddings: {test_embeddings.shape}")
    print(f"   ✅ Test set labels: {test_labels.shape}")

    # =====================================================
    # 4️⃣ Load trained model
    # =====================================================
    print("\n" + "=" * 70)
    print("📥 Loading trained model")
    print("=" * 70)

    try:
        # Initialize model architecture
        model = EmbeddingMLP(
            embedding_dim=embedding_dim,
            hidden_dim_1=512,
            hidden_dim_2=256,
            dropout=0.3
        ).to(device)

        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set to evaluation mode

        print(f"✅ Model loaded successfully!")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # =====================================================
    # 5️⃣ Make predictions
    # =====================================================
    print("\n" + "=" * 70)
    print("🔮 Starting prediction")
    print("=" * 70)

    with torch.no_grad():
        test_X = torch.FloatTensor(test_embeddings).to(device)
        predictions = model(test_X).cpu().numpy()
        actuals = test_labels

    print(f"✅ Prediction completed!")
    print(f"   Number of predicted samples: {len(predictions)}")

    # =====================================================
    # 6️⃣ Calculate evaluation metrics
    # =====================================================
    print("\n" + "=" * 70)
    print("📊 Test Set Evaluation Metrics")
    print("=" * 70)

    metrics = calculate_metrics(predictions, actuals)

    print(f"\n🎯 Independent test set performance:")
    print(f"{'=' * 50}")
    print(f"  MSE (Mean Squared Error):          {metrics['mse']:.6f}")
    print(f"  MAE (Mean Absolute Error):        {metrics['mae']:.6f}")
    print(f"  R² (Coefficient of Determination): {metrics['r2']:.6f}")
    print(f"  Pearson correlation coefficient: {metrics['pearson']:.6f}")
    print(f"{'=' * 50}")

    print("\n" + "=" * 70)
    print("🎉 Test completed!")
    print("=" * 70)

    return metrics


# =====================================================
# 🎬 Main program
# =====================================================
if __name__ == "__main__":

    # =====================================================
    # 📝 Configuration parameters (modify according to your actual setup)
    # =====================================================

    # Model file path
    MODEL_PATH = 'best_model_mol2vec.pth'

    # Test set file path
    TEST_CSV_PATH = r'D:\pycharm\gutingle\pythonProject2\新\test_data.csv'

    # Pretrained Mol2Vec model path
    MOL2VEC_MODEL_PATH = 'model_300dim.pkl'

    # Column names
    SMILES_COLUMN = 'Smiles'
    TARGET_COLUMN = 'pchembl'

    # Mol2Vec parameters
    MOL2VEC_RADIUS = 1

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # =====================================================
    # 🚀 Run test
    # =====================================================

    try:
        metrics = test_model(
            model_path=MODEL_PATH,
            test_csv_path=TEST_CSV_PATH,
            mol2vec_model_path=MOL2VEC_MODEL_PATH,
            smiles_column=SMILES_COLUMN,
            target_column=TARGET_COLUMN,
            mol2vec_radius=MOL2VEC_RADIUS,
            device=DEVICE
        )

        print("\n✅ Program executed successfully!")

    except Exception as e:
        print(f"\n❌ Program execution failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)