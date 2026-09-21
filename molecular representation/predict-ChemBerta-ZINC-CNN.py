import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import sys
import warnings
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')


# =====================================================
# Lazy import transformers
# =====================================================
def lazy_import_transformers():
    """Lazy import transformers to avoid triggering torchvision issues"""
    try:
        from transformers import AutoTokenizer, AutoModel
        return AutoTokenizer, AutoModel
    except Exception as e:
        print(f"❌ Import failed: {e}")
        print("\n💡 Please run the following command to install dependencies:")
        print("   pip install transformers")
        sys.exit(1)


# =====================================================
# CNN model definition (must be exactly the same as during training)
# =====================================================
class EmbeddingCNN(nn.Module):
    """Three-layer CNN for processing embedding vectors with dropout=0.3"""

    def __init__(
            self,
            embedding_dim=768,
            hidden_dim=256,
            cnn_out_channels_1=128,
            cnn_out_channels_2=256,
            cnn_out_channels_3=512,
            kernel_size=3,
            cnn_dropout=0.3,
            fc_dropout=0.3
    ):
        super(EmbeddingCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=cnn_out_channels_1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(cnn_out_channels_1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(cnn_dropout)

        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            in_channels=cnn_out_channels_1,
            out_channels=cnn_out_channels_2,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(cnn_out_channels_2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(cnn_dropout)

        # Third convolutional layer
        self.conv3 = nn.Conv1d(
            in_channels=cnn_out_channels_2,
            out_channels=cnn_out_channels_3,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn3 = nn.BatchNorm1d(cnn_out_channels_3)
        self.pool3 = nn.AdaptiveMaxPool1d(1)
        self.dropout3 = nn.Dropout(cnn_dropout)

        # Fully connected layers
        self.fc1 = nn.Linear(cnn_out_channels_3, hidden_dim)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, embedding_dim)
        x = x.unsqueeze(1)  # (batch, 1, embedding_dim)

        # First convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third convolutional layer
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten
        x = x.squeeze(-1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x.squeeze(-1)


# =====================================================
# Data extraction function
# =====================================================
def extract_embeddings(df, smiles_column, tokenizer, model, device, batch_size=16):
    """Extract embedding vectors from DataFrame"""

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

    all_embeddings = []

    for i in tqdm(range(0, len(smiles_list), batch_size), desc="   Extracting embeddings"):
        batch = smiles_list[i:i + batch_size]

        try:
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                cls_embeddings = cls_embeddings.cpu().numpy()
                all_embeddings.append(cls_embeddings)

        except Exception as e:
            print(f"\n   ⚠️  Batch processing failed: {e}")
            embedding_dim = model.config.hidden_size
            zero_embeddings = np.zeros((len(batch), embedding_dim))
            all_embeddings.append(zero_embeddings)

    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings, df


# =====================================================
# Calculate evaluation metrics
# =====================================================
def calculate_metrics(predictions, actuals):
    """Calculate regression evaluation metrics"""
    # MSE
    mse = np.mean((predictions - actuals) ** 2)

    # RMSE
    rmse = np.sqrt(mse)

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
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr,
        'pearson_pvalue': p_value
    }


# =====================================================
# Environment configuration
# =====================================================
def setup_environment():
    """Configure environment and mirror"""
    print("🌐 Configuring download environment...")
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    print("✅ Mirror acceleration enabled")


# =====================================================
# Main program
# =====================================================
if __name__ == "__main__":
    # =====================================================
    # 1. Parameter settings
    # =====================================================
    # File paths
    test_csv_path = r"test_data.csv"  # Change to your test dataset path
    model_path = "best_model.pth"  # Best model weight file path

    # Column names
    smiles_column = "Smiles"
    target_column = "pchembl"

    # Model parameters (must be consistent with training)
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    batch_size_embedding = 16

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("🧪 ChemBERTa-ZINC + CNN Independent Test Set Evaluation")
    print("=" * 70)
    print(f"🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"📦 PyTorch version: {torch.__version__}")
    print(f"📦 Pretrained model: {model_name}")
    print(f"📁 Model weights: {model_path}")
    print(f"📁 Test dataset: {test_csv_path}")
    print("=" * 70)

    # =====================================================
    # 2. Check whether the model file exists
    # =====================================================
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model file '{model_path}' not found")
        print("💡 Please make sure the model has been trained and the weight file has been saved")
        sys.exit(1)

    # =====================================================
    # 3. Read test dataset
    # =====================================================
    print("\n📂 Reading test dataset...")

    try:
        test_df = pd.read_csv(test_csv_path, encoding='utf-8')
        print(f"✅ Test dataset: {len(test_df)} rows")
    except Exception as e:
        print(f"❌ Failed to read test dataset: {e}")
        sys.exit(1)

    # Check whether columns exist
    if smiles_column not in test_df.columns:
        print(f"❌ Column '{smiles_column}' does not exist in the test dataset")
        print(f"💡 Available columns: {', '.join(test_df.columns.tolist())}")
        sys.exit(1)
    if target_column not in test_df.columns:
        print(f"❌ Column '{target_column}' does not exist in the test dataset")
        print(f"💡 Available columns: {', '.join(test_df.columns.tolist())}")
        sys.exit(1)

    # =====================================================
    # 4. Load ChemBERTa-ZINC model
    # =====================================================
    print("\n" + "=" * 70)
    print("📥 Loading ChemBERTa-ZINC model")
    print("=" * 70)

    setup_environment()

    # Lazy import transformers
    print("📦 Loading transformers library...")
    TokenizerClass, ModelClass = lazy_import_transformers()
    print("✅ transformers imported successfully")

    # Load model
    print(f"\n📦 Preparing to load model: {model_name}")

    try:
        print("   📥 Loading tokenizer...")
        tokenizer = TokenizerClass.from_pretrained(model_name, trust_remote_code=True)

        print("   📥 Loading model...")
        chemberta_model = ModelClass.from_pretrained(model_name, trust_remote_code=True)

        chemberta_model = chemberta_model.to(device)
        chemberta_model.eval()

        embedding_dim = chemberta_model.config.hidden_size
        print(f"✅ Model loaded successfully!")
        print(f"📐 Embedding dimension: {embedding_dim}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\n💡 Please check your network connection or try downloading the model manually")
        sys.exit(1)

    # =====================================================
    # 5. Extract test set embedding vectors
    # =====================================================
    print("\n" + "=" * 70)
    print("🔄 Extracting test set embeddings")
    print("=" * 70)

    print("\n📊 Processing test dataset...")
    test_embeddings, test_df_clean = extract_embeddings(
        test_df, smiles_column, tokenizer, chemberta_model, device, batch_size_embedding
    )
    test_labels = test_df_clean[target_column].values

    print(f"   ✅ Test set embeddings: {test_embeddings.shape}")

    # Clear memory
    if device == "cuda":
        del chemberta_model
        torch.cuda.empty_cache()

    # =====================================================
    # 6. Load trained CNN model
    # =====================================================
    print("\n" + "=" * 70)
    print("🏗️  Loading trained CNN model")
    print("=" * 70)

    cnn_model = EmbeddingCNN(
        embedding_dim=embedding_dim,
        hidden_dim=256,
        cnn_out_channels_1=128,
        cnn_out_channels_2=256,
        cnn_out_channels_3=512,
        kernel_size=3,
        cnn_dropout=0.3,
        fc_dropout=0.3
    ).to(device)

    # Load model weights
    try:
        cnn_model.load_state_dict(torch.load(model_path, map_location=device))
        cnn_model.eval()
        print(f"✅ Successfully loaded model weights: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        sys.exit(1)

    # Print model parameters
    total_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"📊 Total model parameters: {total_params:,}")

    # =====================================================
    # 7. Prediction and evaluation on test set
    # =====================================================
    print("\n" + "=" * 70)
    print("📊 Test Set Evaluation")
    print("=" * 70)

    cnn_model.eval()
    with torch.no_grad():
        test_X = torch.FloatTensor(test_embeddings).to(device)
        predictions = cnn_model(test_X).cpu().numpy()
        actual = test_labels

        # Calculate evaluation metrics
        test_metrics = calculate_metrics(predictions, actual)

        print(f"\n🎯 Test set performance:")
        print(f"   MSE:            {test_metrics['mse']:.6f}")
        print(f"   RMSE:           {test_metrics['rmse']:.6f}")
        print(f"   MAE:            {test_metrics['mae']:.6f}")
        print(f"   Pearson:        {test_metrics['pearson']:.6f}")
        print(f"   Pearson p-value:{test_metrics['pearson_pvalue']:.6e}")
        print(f"   R²:             {test_metrics['r2']:.6f}")

    # =====================================================
    # 8. Save prediction results
    # =====================================================
    results_df = test_df_clean.copy()
    results_df['Predicted'] = predictions
    results_df['Actual'] = actual
    results_df['Error'] = predictions - actual
    results_df['Abs_Error'] = np.abs(predictions - actual)

    output_file = 'test_predictions.csv'
    results_df.to_csv(output_file, index=False)

    print(f"\n💾 Prediction results saved to: {output_file}")

    # =====================================================
    # 9. Output statistical summary
    # =====================================================
    print("\n" + "=" * 70)
    print("📈 Prediction Statistical Summary")
    print("=" * 70)
    print(f"Number of samples:       {len(predictions)}")
    print(f"Actual value range:      [{actual.min():.4f}, {actual.max():.4f}]")
    print(f"Predicted value range:   [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Mean absolute error:     {test_metrics['mae']:.6f}")
    print(f"Root mean squared error: {test_metrics['rmse']:.6f}")
    print(f"Maximum error:           {np.max(np.abs(predictions - actual)):.6f}")
    print(f"Minimum error:           {np.min(np.abs(predictions - actual)):.6f}")

    print("\n" + "=" * 70)
    print("🎉 Test completed!")
    print("=" * 70)