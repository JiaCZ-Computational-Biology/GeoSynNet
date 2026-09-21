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
# LSTM model definition (unidirectional) - must be exactly the same as during training
# =====================================================
class EmbeddingLSTM(nn.Module):
    """Three-layer unidirectional LSTM for embedding vectors; dropout=0.3 is used in the LSTM and before the FC layer"""

    def __init__(
            self,
            embedding_dim=768,
            hidden_dim=256,
            lstm_hidden_1=128,
            lstm_hidden_2=256,
            lstm_hidden_3=512,
            lstm_dropout=0.3,
            fc_dropout=0.3
    ):
        super(EmbeddingLSTM, self).__init__()

        # Layer 1
        self.lstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_1,
            num_layers=1,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=False
        )
        self.bn1 = nn.BatchNorm1d(lstm_hidden_1)

        # Layer 2
        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden_1,
            hidden_size=lstm_hidden_2,
            num_layers=1,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=False
        )
        self.bn2 = nn.BatchNorm1d(lstm_hidden_2)

        # Layer 3
        self.lstm3 = nn.LSTM(
            input_size=lstm_hidden_2,
            hidden_size=lstm_hidden_3,
            num_layers=1,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=False
        )
        self.bn3 = nn.BatchNorm1d(lstm_hidden_3)

        # Fully connected layers
        fc_in = lstm_hidden_3
        self.fc1 = nn.Linear(fc_in, hidden_dim)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)

        # Layer 1
        x, _ = self.lstm1(x)
        x = x.squeeze(1)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.unsqueeze(1)

        # Layer 2
        x, _ = self.lstm2(x)
        x = x.squeeze(1)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.unsqueeze(1)

        # Layer 3
        x, _ = self.lstm3(x)
        x = x.squeeze(1)
        x = self.bn3(x)
        x = self.relu(x)

        # FC
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
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
        'p_value': p_value
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
# Main program - test set evaluation
# =====================================================
if __name__ == "__main__":

    print("=" * 70)
    print("🧪 ChemBERTa-ZINC + LSTM Independent Test Set Evaluation")
    print("=" * 70)

    # =====================================================
    # 1️⃣ Parameter settings
    # =====================================================
    # File paths
    test_csv_path = r"test_data.csv"  # Change to your test dataset path
    model_path = "best_model.pth"  # Trained model path

    # Column names (must be consistent with training)
    smiles_column = "Smiles"
    target_column = "pchembl"

    # Model parameters (must be exactly the same as during training)
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    embedding_dim = 768  # ChemBERTa-ZINC embedding dimension
    batch_size_embedding = 16

    # LSTM model parameters (must be exactly the same as during training)
    hidden_dim = 256
    lstm_hidden_1 = 128
    lstm_hidden_2 = 256
    lstm_hidden_3 = 512
    lstm_dropout = 0.3
    fc_dropout = 0.3

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📦 Pretrained model: {model_name}")
    print(f"📂 Model weights: {model_path}")
    print("=" * 70)

    # =====================================================
    # 2️⃣ Check whether files exist
    # =====================================================
    if not os.path.exists(test_csv_path):
        print(f"❌ Error: Test dataset file does not exist")
        print(f"   Path: {test_csv_path}")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file does not exist")
        print(f"   Path: {model_path}")
        print(f"💡 Please run the training code first to generate best_model.pth")
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
    # 4️⃣ Load ChemBERTa-ZINC model
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

        print(f"✅ ChemBERTa model loaded successfully!")
        print(f"📐 Embedding dimension: {chemberta_model.config.hidden_size}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # =====================================================
    # 5️⃣ Extract test set embeddings
    # =====================================================
    print("\n" + "=" * 70)
    print("🔄 Extracting test set embeddings")
    print("=" * 70)

    test_embeddings, test_df_clean = extract_embeddings(
        test_df, smiles_column, tokenizer, chemberta_model, device, batch_size_embedding
    )
    test_labels = test_df_clean[target_column].values

    print(f"   ✅ Test set embeddings: {test_embeddings.shape}")
    print(f"   ✅ Test set labels: {test_labels.shape}")

    # Clear memory
    if device == "cuda":
        del chemberta_model
        torch.cuda.empty_cache()

    # =====================================================
    # 6️⃣ Load trained LSTM model
    # =====================================================
    print("\n" + "=" * 70)
    print("🏗️  Loading trained LSTM model")
    print("=" * 70)

    # Create model instance (parameters must be consistent with training)
    lstm_model = EmbeddingLSTM(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        lstm_hidden_1=lstm_hidden_1,
        lstm_hidden_2=lstm_hidden_2,
        lstm_hidden_3=lstm_hidden_3,
        lstm_dropout=lstm_dropout,
        fc_dropout=fc_dropout
    ).to(device)

    # Load trained weights
    try:
        lstm_model.load_state_dict(torch.load(model_path, map_location=device))
        lstm_model.eval()
        print(f"✅ Successfully loaded model weights: {model_path}")

        # Print model parameter count
        total_params = sum(p.numel() for p in lstm_model.parameters())
        print(f"📊 Model parameters: {total_params:,}")

    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        sys.exit(1)

    # =====================================================
    # 7️⃣ Predict on test set
    # =====================================================
    print("\n" + "=" * 70)
    print("🔮 Test Set Prediction")
    print("=" * 70)

    with torch.no_grad():
        test_X = torch.FloatTensor(test_embeddings).to(device)
        predictions = lstm_model(test_X).cpu().numpy()
        actuals = test_labels

    print(f"✅ Prediction completed")
    print(f"   Predicted value range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"   Actual value range: [{actuals.min():.4f}, {actuals.max():.4f}]")

    # =====================================================
    # 8️⃣ Calculate evaluation metrics
    # =====================================================
    print("\n" + "=" * 70)
    print("📊 Test Set Evaluation Metrics")
    print("=" * 70)

    metrics = calculate_metrics(predictions, actuals)

    print(f"\n🎯 Independent test set performance:")
    print(f"   MSE (Mean Squared Error):          {metrics['mse']:.6f}")
    print(f"   RMSE (Root Mean Squared Error):   {metrics['rmse']:.6f}")
    print(f"   MAE (Mean Absolute Error):        {metrics['mae']:.6f}")
    print(f"   R² (Coefficient of Determination): {metrics['r2']:.6f}")
    print(f"   Pearson correlation coefficient: {metrics['pearson']:.6f}")
    print(f"   Pearson p-value:                 {metrics['p_value']:.6e}")

    # =====================================================
    # 9️⃣ Save prediction results
    # =====================================================
    print("\n" + "=" * 70)
    print("💾 Saving prediction results")
    print("=" * 70)

    # Create result DataFrame
    results_df = test_df_clean.copy()
    results_df['Predicted'] = predictions
    results_df['Actual'] = actuals
    results_df['Error'] = predictions - actuals
    results_df['Abs_Error'] = np.abs(predictions - actuals)
    results_df['Squared_Error'] = (predictions - actuals) ** 2

    # Save detailed results
    output_path = 'test_predictions.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Detailed prediction results saved to: {output_path}")

    # Save evaluation metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = 'test_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8')
    print(f"✅ Evaluation metrics saved to: {metrics_path}")

    # =====================================================
    # 🔟 Statistical analysis
    # =====================================================
    print("\n" + "=" * 70)
    print("📈 Error Statistical Analysis")
    print("=" * 70)

    errors = predictions - actuals
    abs_errors = np.abs(errors)

    print(f"\nError statistics:")
    print(f"   Mean Error (ME):              {np.mean(errors):.6f}")
    print(f"   Error standard deviation:     {np.std(errors):.6f}")
    print(f"   Maximum positive error:       {np.max(errors):.6f}")
    print(f"   Maximum negative error:       {np.min(errors):.6f}")
    print(f"   Median absolute error:        {np.median(abs_errors):.6f}")

    # Calculate accuracy at different thresholds
    thresholds = [0.5, 1.0, 1.5, 2.0]
    print(f"\nPrediction accuracy (proportion with absolute error below threshold):")
    for threshold in thresholds:
        accuracy = np.mean(abs_errors < threshold) * 100
        print(f"   Threshold {threshold}: {accuracy:.2f}%")

    print("\n" + "=" * 70)
    print("🎉 Test set evaluation completed!")
    print("=" * 70)