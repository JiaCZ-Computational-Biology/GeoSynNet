import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import sys
import warnings
from scipy.stats import pearsonr
import random
from rdkit import Chem
from rdkit.Chem import AllChem
from gensim.models import word2vec
import urllib.request

warnings.filterwarnings('ignore')


# =====================================================
# 🌱 Set random seed
# =====================================================
def set_seed(seed=42):
    """Set all random seeds to ensure reproducibility"""
    print(f"🌱 Setting random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("✅ Random seed set successfully")


# =====================================================
# 📥 Download and load Mol2Vec model
# =====================================================
def download_mol2vec_model():
    """Download the pretrained Mol2Vec model"""
    model_path = 'model_300dim.pkl'
    if not os.path.exists(model_path):
        print("📥 Downloading the pretrained Mol2Vec model (about 20 MB)...")
        url = 'https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl'
        try:
            urllib.request.urlretrieve(url, model_path)
            print("✅ Model download completed!")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("💡 Please download the model manually and place it in the current directory")
            raise
    else:
        print(f"✅ Local model found: {model_path}")
    return model_path


def load_mol2vec_model():
    """Load Mol2Vec model"""
    try:
        model_path = download_mol2vec_model()
        model = word2vec.Word2Vec.load(model_path)

        # Check Gensim version
        import gensim
        print(f"📦 Gensim version: {gensim.__version__}")

        # Get vocabulary size (compatible with Gensim 3.x and 4.x)
        vocab_size = len(model.wv.key_to_index) if hasattr(model.wv, 'key_to_index') else len(model.wv.vocab)
        vector_dim = model.wv.vector_size

        print(f"✅ Mol2Vec model loaded successfully!")
        print(f"   Vocabulary size: {vocab_size:,}")
        print(f"   Vector dimension: {vector_dim}")

        return model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        sys.exit(1)


# =====================================================
# 🧪 Mol2Vec conversion function
# =====================================================
def mol2alt_sentence_fixed(mol, radius=1):
    """
    Convert a molecule into a "sentence" (list of substructure identifiers)
    Compatible with Gensim 4.x
    """
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
    """
    Convert a list of sentences into sequence vectors (for CNN)
    Return padded sequences and actual lengths
    Optimize memory usage by using float32 instead of float64
    """
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
            # Fill empty sentences with zero vectors using float32
            sequences.append(np.zeros((1, model.wv.vector_size), dtype=np.float32))
            lengths.append(1)
        else:
            # Truncate or retain original length
            sentence = sentence[:max_len]
            word_vectors = [get_vector(word) for word in sentence]
            # Convert to float32 to save memory
            sequences.append(np.array(word_vectors, dtype=np.float32))
            lengths.append(len(word_vectors))

    # Pad to a uniform length
    padded_sequences = []
    actual_max_len = min(max(lengths), max_len)

    for seq in sequences:
        if len(seq) < actual_max_len:
            # Use float32
            padding = np.zeros((actual_max_len - len(seq), model.wv.vector_size), dtype=np.float32)
            padded_seq = np.vstack([seq, padding])
        else:
            padded_seq = seq[:actual_max_len]
        padded_sequences.append(padded_seq)

    # Return float32 array
    return np.array(padded_sequences, dtype=np.float32), np.array(lengths, dtype=np.int32)


# =====================================================
# 📊 Extract sequence embeddings using Mol2Vec
# =====================================================
def extract_mol2vec_sequences(df, smiles_column, model, radius=1, max_len=100):
    """Extract Mol2Vec sequence embeddings from a DataFrame (for CNN)"""

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
    lengths = []
    valid_indices = []
    failed_count = 0

    print("   🔄 Starting molecule conversion...")
    for idx, smiles in enumerate(tqdm(smiles_list, desc="   Extracting Mol2Vec sequences")):
        try:
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                failed_count += 1
                continue

            # Convert to sentence
            sentence = mol2alt_sentence_fixed(mol, radius=radius)
            sequences.append(sentence)
            valid_indices.append(idx)

        except Exception as e:
            failed_count += 1
            if failed_count <= 3:  # Only print the first three errors
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
    print(f"   📏 Data type: {padded_sequences.dtype}")
    print(f"   📏 Memory usage: {padded_sequences.nbytes / (1024**2):.2f} MB")
    print(f"   📏 Average sequence length: {np.mean(seq_lengths):.1f}")
    print(f"   📏 Maximum sequence length: {np.max(seq_lengths)}")

    return padded_sequences, seq_lengths, df_clean


# =====================================================
# 🏗️ Three-layer CNN model definition
# =====================================================
class ThreeLayerCNN(nn.Module):
    """Three-layer CNN for processing sequence embeddings; dropout=0.3"""

    def __init__(
            self,
            input_dim=300,  # Mol2Vec is 300-dimensional by default
            num_filters_1=256,
            num_filters_2=128,
            num_filters_3=64,
            kernel_size=3,
            dropout=0.3
    ):
        super(ThreeLayerCNN, self).__init__()

        # First Conv1d layer
        # Input: (batch, input_dim, seq_len)
        # Output: (batch, num_filters_1, seq_len)
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_filters_1,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # Keep sequence length unchanged
        )
        self.bn1 = nn.BatchNorm1d(num_filters_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Halve the length

        # Second Conv1d layer
        self.conv2 = nn.Conv1d(
            in_channels=num_filters_1,
            out_channels=num_filters_2,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(num_filters_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Halve the length again

        # Third Conv1d layer
        self.conv3 = nn.Conv1d(
            in_channels=num_filters_2,
            out_channels=num_filters_3,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn3 = nn.BatchNorm1d(num_filters_3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected output layer
        self.fc = nn.Linear(num_filters_3, 1)

    def forward(self, x, lengths=None):
        # x shape: (batch, seq_len, input_dim)
        # Required CNN format: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)

        # First CNN layer
        x = self.conv1(x)  # (batch, num_filters_1, seq_len)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.pool1(x)  # (batch, num_filters_1, seq_len/2)

        # Second CNN layer
        x = self.conv2(x)  # (batch, num_filters_2, seq_len/2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.pool2(x)  # (batch, num_filters_2, seq_len/4)

        # Third CNN layer
        x = self.conv3(x)  # (batch, num_filters_3, seq_len/4)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # Global average pooling
        x = self.global_avg_pool(x)  # (batch, num_filters_3, 1)
        x = x.squeeze(-1)  # (batch, num_filters_3)

        # Output layer
        output = self.fc(x)  # (batch, 1)
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
# 🎯 Training function
# =====================================================
def train_model(model, train_sequences, train_lengths, train_labels,
                val_sequences, val_lengths, val_labels,
                device, epochs=100, batch_size=32, learning_rate=0.001, seed=42):
    """Train the CNN model"""

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    # Convert to tensors
    train_X = torch.FloatTensor(train_sequences).to(device)
    train_len = torch.LongTensor(train_lengths)
    train_y = torch.FloatTensor(train_labels).to(device)

    val_X = torch.FloatTensor(val_sequences).to(device)
    val_len = torch.LongTensor(val_lengths)
    val_y = torch.FloatTensor(val_labels).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_mse = float('inf')
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = 50

    print("\n" + "=" * 70)
    print("🚀 Starting training")
    print("=" * 70)

    for epoch in range(epochs):
        # Training mode
        model.train()
        train_losses = []

        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed + epoch)
        indices = torch.randperm(len(train_X), generator=generator)

        for i in range(0, len(train_X), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_X = train_X[batch_indices]
            batch_len = train_len[batch_indices]
            batch_y = train_y[batch_indices]

            optimizer.zero_grad()
            outputs = model(batch_X, batch_len)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            train_losses.append(loss.item())

        # Validation mode
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X, val_len)
            val_predictions = val_outputs.cpu().numpy()
            val_actuals = val_y.cpu().numpy()
            val_metrics = calculate_metrics(val_predictions, val_actuals)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['mse'])
        new_lr = optimizer.param_groups[0]['lr']

        train_mse = np.mean(train_losses)

        lr_info = f"LR: {new_lr:.6f}"
        if new_lr != old_lr:
            lr_info += f" ⬇️ (reduced from {old_lr:.6f})"

        is_best = ""
        if val_metrics['mse'] < best_val_mse:
            is_best = " ⭐ New best!"
            best_val_mse = val_metrics['mse']
            best_metrics = val_metrics.copy()
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_cnn.pth')
        else:
            patience_counter += 1

        print(f"\nEpoch [{epoch + 1:3d}/{epochs}]{is_best}")
        print(f"  Train MSE:   {train_mse:.6f}")
        print(f"  Val MSE:     {val_metrics['mse']:.6f}")
        print(f"  Val RMSE:    {val_metrics['rmse']:.6f}")
        print(f"  Val MAE:     {val_metrics['mae']:.6f}")
        print(f"  Val Pearson: {val_metrics['pearson']:.6f}")
        print(f"  Val R²:      {val_metrics['r2']:.6f}")
        print(f"  {lr_info}")

        if patience_counter >= early_stop_patience:
            print(f"\n⚠️  Early stopping at epoch {epoch + 1}")
            print(f"   Validation MSE has not improved for {early_stop_patience} consecutive epochs")
            break

    print("\n" + "=" * 70)
    print("✅ Training completed!")
    print(f"🏆 Best validation metrics (Epoch {best_epoch}):")
    print(f"   MSE:     {best_metrics['mse']:.6f}")
    print(f"   RMSE:    {best_metrics['rmse']:.6f}")
    print(f"   MAE:     {best_metrics['mae']:.6f}")
    print(f"   Pearson: {best_metrics['pearson']:.6f}")
    print(f"   R²:      {best_metrics['r2']:.6f}")
    print("=" * 70)

    model.load_state_dict(torch.load('best_model_cnn.pth'))
    return best_metrics


# =====================================================
# 🎬 Main program
# =====================================================
if __name__ == "__main__":
    # =====================================================
    # 🌱 Set random seed
    # =====================================================
    RANDOM_SEED = 42
    set_seed(RANDOM_SEED)

    # =====================================================
    # 1️⃣ Parameter settings
    # =====================================================
    train_csv_path = r"D:\pycharm\gutingle\pythonProject2\新\train_data.csv"
    val_csv_path = r"D:\pycharm\gutingle\pythonProject2\新\validation_data.csv"

    smiles_column = "Smiles"
    target_column = "pchembl"

    mol2vec_radius = 1  # Mol2Vec radius parameter
    max_sequence_length = 100  # Maximum CNN sequence length
    batch_size_training = 32
    epochs = 1000
    learning_rate = 0.001

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("🧬 Mol2Vec + Three-Layer CNN Molecular Property Prediction")
    print("=" * 70)
    print(f"🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"📦 PyTorch version: {torch.__version__}")
    print(f"🌱 Random seed: {RANDOM_SEED}")
    print("=" * 70)

    # =====================================================
    # 2️⃣ Read data
    # =====================================================
    print("\n📂 Reading datasets...")

    try:
        train_df = pd.read_csv(train_csv_path, encoding='utf-8')
        print(f"✅ Training set: {len(train_df)} rows")
    except Exception as e:
        print(f"❌ Failed to read training set: {e}")
        sys.exit(1)

    try:
        val_df = pd.read_csv(val_csv_path, encoding='utf-8')
        print(f"✅ Validation set: {len(val_df)} rows")
    except Exception as e:
        print(f"❌ Failed to read validation set: {e}")
        sys.exit(1)

    # Check whether columns exist
    for df, name in [(train_df, "training set"), (val_df, "validation set")]:
        if smiles_column not in df.columns:
            print(f"❌ Column '{smiles_column}' does not exist in the {name}")
            print(f"💡 Available columns: {', '.join(df.columns.tolist())}")
            sys.exit(1)
        if target_column not in df.columns:
            print(f"❌ Column '{target_column}' does not exist in the {name}")
            print(f"💡 Available columns: {', '.join(df.columns.tolist())}")
            sys.exit(1)

    # =====================================================
    # 3️⃣ Load Mol2Vec model
    # =====================================================
    print("\n" + "=" * 70)
    print("📥 Loading Mol2Vec model")
    print("=" * 70)

    mol2vec_model = load_mol2vec_model()
    embedding_dim = mol2vec_model.wv.vector_size
    print(f"📐 Embedding dimension: {embedding_dim}")

    # =====================================================
    # 4️⃣ Extract sequence embeddings
    # =====================================================
    print("\n" + "=" * 70)
    print("🔄 Extracting Mol2Vec sequence embeddings")
    print("=" * 70)

    print("\n📊 Processing training set...")
    train_sequences, train_lengths, train_df_clean = extract_mol2vec_sequences(
        train_df, smiles_column, mol2vec_model,
        radius=mol2vec_radius, max_len=max_sequence_length
    )
    train_labels = train_df_clean[target_column].values
    print(f"   ✅ Training set sequences: {train_sequences.shape}")

    print("\n📊 Processing validation set...")
    val_sequences, val_lengths, val_df_clean = extract_mol2vec_sequences(
        val_df, smiles_column, mol2vec_model,
        radius=mol2vec_radius, max_len=max_sequence_length
    )
    val_labels = val_df_clean[target_column].values
    print(f"   ✅ Validation set sequences: {val_sequences.shape}")

    # =====================================================
    # 5️⃣ Build and train CNN model
    # =====================================================
    print("\n" + "=" * 70)
    print("🏗️  Building three-layer CNN model")
    print("=" * 70)

    cnn_model = ThreeLayerCNN(
        input_dim=embedding_dim,
        num_filters_1=256,
        num_filters_2=128,
        num_filters_3=64,
        kernel_size=3,
        dropout=0.3
    ).to(device)

    total_params = sum(p.numel() for p in cnn_model.parameters())
    trainable_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
    print(f"\n📊 Model parameters:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"\n🏗️  Model architecture:")
    print(f"   Input dimension:     {embedding_dim}")
    print(f"   CNN layer 1:         {256} filters (kernel_size=3) + MaxPool(2)")
    print(f"   CNN layer 2:         {128} filters (kernel_size=3) + MaxPool(2)")
    print(f"   CNN layer 3:         {64} filters (kernel_size=3)")
    print(f"   Global average pooling")
    print(f"   Fully connected:     {64} -> 1")
    print(f"   Dropout:             0.3")

    best_metrics = train_model(
        cnn_model,
        train_sequences,
        train_lengths,
        train_labels,
        val_sequences,
        val_lengths,
        val_labels,
        device,
        epochs=epochs,
        batch_size=batch_size_training,
        learning_rate=learning_rate,
        seed=RANDOM_SEED
    )

    # =====================================================
    # 6️⃣ Final evaluation
    # =====================================================
    print("\n" + "=" * 70)
    print("📊 Final evaluation")
    print("=" * 70)

    cnn_model.eval()
    with torch.no_grad():
        val_X = torch.FloatTensor(val_sequences).to(device)
        val_len = torch.LongTensor(val_lengths)
        predictions = cnn_model(val_X, val_len).cpu().numpy()
        actual = val_labels

        final_metrics = calculate_metrics(predictions, actual)

        print(f"🎯 Validation set performance:")
        print(f"   MSE:     {final_metrics['mse']:.6f}")
        print(f"   RMSE:    {final_metrics['rmse']:.6f}")
        print(f"   MAE:     {final_metrics['mae']:.6f}")
        print(f"   Pearson: {final_metrics['pearson']:.6f}")
        print(f"   R²:      {final_metrics['r2']:.6f}")

    # Save prediction results
    results_df = val_df_clean.copy()
    results_df['Predicted'] = predictions
    results_df['Actual'] = actual
    results_df['Error'] = predictions - actual
    results_df['Abs_Error'] = np.abs(predictions - actual)
    results_df.to_csv('predictions_cnn.csv', index=False)

    print(f"\n💾 Prediction results saved to: predictions_cnn.csv")
    print(f"💾 Best model saved to: best_model_cnn.pth")
    print(f"🌱 Random seed used: {RANDOM_SEED}")

    print("\n" + "=" * 70)
    print("🎉 All tasks completed!")
    print("=" * 70)