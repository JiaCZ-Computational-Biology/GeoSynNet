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
# BiLSTM Model Definition
# =====================================================
class EmbeddingBiLSTM(nn.Module):
    """Three-layer bidirectional LSTM for embedding vectors with dropout=0.3."""

    def __init__(
        self,
        embedding_dim=768,  # Default ChemBERTa-ZINC embedding dimension
        hidden_dim=256,
        lstm_hidden_1=128,
        lstm_hidden_2=256,
        lstm_hidden_3=512,
        lstm_dropout=0.3,
        fc_dropout=0.3
    ):
        super(EmbeddingBiLSTM, self).__init__()

        # Layer 1: input=embedding_dim -> hidden=128
        # Bidirectional output dimension = 256
        self.lstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_1,
            num_layers=1,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True
        )
        self.bn1 = nn.BatchNorm1d(lstm_hidden_1 * 2)

        # Layer 2: input=256 -> hidden=256
        # Bidirectional output dimension = 512
        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden_1 * 2,
            hidden_size=lstm_hidden_2,
            num_layers=1,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True
        )
        self.bn2 = nn.BatchNorm1d(lstm_hidden_2 * 2)

        # Layer 3: input=512 -> hidden=512
        # Bidirectional output dimension = 1024
        self.lstm3 = nn.LSTM(
            input_size=lstm_hidden_2 * 2,
            hidden_size=lstm_hidden_3,
            num_layers=1,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True
        )
        self.bn3 = nn.BatchNorm1d(lstm_hidden_3 * 2)

        # Fully connected layers
        fc_in = lstm_hidden_3 * 2
        self.fc1 = nn.Linear(fc_in, hidden_dim)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, embedding_dim)
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

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze(-1)


# =====================================================
# Lazy Import of Transformers
# =====================================================
def lazy_import_transformers():
    """Lazily import transformers."""

    try:
        from transformers import AutoTokenizer, AutoModel
        return AutoTokenizer, AutoModel

    except Exception as e:
        print(f"❌ Import failed: {e}")
        print("\n💡 Please run the following command to install dependencies:")
        print("   pip install transformers")
        sys.exit(1)


# =====================================================
# Embedding Extraction Function
# =====================================================
def extract_embeddings(
    df,
    smiles_column,
    tokenizer,
    model,
    device,
    batch_size=16
):
    """Extract embedding vectors from a DataFrame."""

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

    for i in tqdm(
        range(0, len(smiles_list), batch_size),
        desc="   Extracting embeddings"
    ):
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

            zero_embeddings = np.zeros(
                (len(batch), embedding_dim)
            )

            all_embeddings.append(zero_embeddings)

    all_embeddings = np.vstack(all_embeddings)

    return all_embeddings, df


# =====================================================
# Evaluation Metrics
# =====================================================
def calculate_metrics(predictions, actuals):
    """Calculate regression evaluation metrics."""

    # Mean Squared Error
    mse = np.mean(
        (predictions - actuals) ** 2
    )

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.mean(
        np.abs(predictions - actuals)
    )

    # R-squared score
    ss_res = np.sum(
        (actuals - predictions) ** 2
    )

    ss_tot = np.sum(
        (actuals - np.mean(actuals)) ** 2
    )

    r2 = (
        1 - (ss_res / ss_tot)
        if ss_tot != 0
        else 0
    )

    # Pearson correlation coefficient
    try:
        pearson_corr, p_value = pearsonr(
            predictions,
            actuals
        )

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
# Environment Configuration
# =====================================================
def setup_environment():
    """Configure the model download environment."""

    print("🌐 Configuring download environment...")

    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

    print("✅ Mirror acceleration enabled")


# =====================================================
# Main Program
# =====================================================
if __name__ == "__main__":

    # =====================================================
    # Parameter Settings
    # =====================================================

    # File paths
    test_csv_path = (
        r"test_data.csv"
    )

    model_path = "best_model.pth"

    # Column names
    smiles_column = "Smiles"

    target_column = "pchembl"

    # Model parameters
    model_name = "seyonec/ChemBERTa-zinc-base-v1"

    batch_size_embedding = 16

    # Device configuration
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)

    print(
        "🧪 ChemBERTa-ZINC + BiLSTM "
        "Independent Test Set Evaluation"
    )

    print("=" * 70)

    print(f"🔧 Device: {device}")

    if torch.cuda.is_available():

        print(
            f"🎮 GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"📦 Pretrained model: "
        f"{model_name}"
    )

    print(
        f"💾 Model checkpoint: "
        f"{model_path}"
    )

    print("=" * 70)


    # =====================================================
    # Check Model File
    # =====================================================
    if not os.path.exists(model_path):

        print(
            f"❌ Model file does not exist: "
            f"{model_path}"
        )

        print(
            "💡 Please run the training script "
            "first to generate the model file"
        )

        sys.exit(1)


    # =====================================================
    # Load Test Dataset
    # =====================================================
    print(
        "\n📂 Loading test dataset..."
    )

    try:
        test_df = pd.read_csv(
            test_csv_path,
            encoding='utf-8'
        )

        print(
            f"✅ Test dataset: "
            f"{len(test_df)} rows"
        )

    except Exception as e:

        print(
            f"❌ Failed to load test dataset: "
            f"{e}"
        )

        sys.exit(1)


    # Check required columns
    if smiles_column not in test_df.columns:

        print(
            f"❌ Column '{smiles_column}' "
            f"does not exist in the test dataset"
        )

        print(
            f"💡 Available columns: "
            f"{', '.join(test_df.columns.tolist())}"
        )

        sys.exit(1)


    if target_column not in test_df.columns:

        print(
            f"❌ Column '{target_column}' "
            f"does not exist in the test dataset"
        )

        print(
            f"💡 Available columns: "
            f"{', '.join(test_df.columns.tolist())}"
        )

        sys.exit(1)


    # =====================================================
    # Load ChemBERTa-ZINC Model
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "📥 Loading ChemBERTa-ZINC model"
    )

    print("=" * 70)

    setup_environment()


    # Lazy import of transformers
    print(
        "📦 Loading transformers library..."
    )

    TokenizerClass, ModelClass = (
        lazy_import_transformers()
    )

    print(
        "✅ Transformers imported successfully"
    )


    # Load pretrained model
    try:

        print(
            "   📥 Loading tokenizer..."
        )

        tokenizer = (
            TokenizerClass.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        )

        print(
            "   📥 Loading pretrained model..."
        )

        chemberta_model = (
            ModelClass.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        )

        chemberta_model = (
            chemberta_model.to(device)
        )

        chemberta_model.eval()

        embedding_dim = (
            chemberta_model.config.hidden_size
        )

        print(
            "✅ Pretrained model loaded successfully"
        )

        print(
            f"📐 Embedding dimension: "
            f"{embedding_dim}"
        )

    except Exception as e:

        print(
            f"❌ Failed to load pretrained model: "
            f"{e}"
        )

        sys.exit(1)


    # =====================================================
    # Extract Test Embeddings
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "🔄 Extracting test set embeddings"
    )

    print("=" * 70)

    test_embeddings, test_df_clean = (
        extract_embeddings(
            test_df,
            smiles_column,
            tokenizer,
            chemberta_model,
            device,
            batch_size_embedding
        )
    )

    test_labels = (
        test_df_clean[target_column].values
    )

    print(
        f"   ✅ Test embeddings: "
        f"{test_embeddings.shape}"
    )

    print(
        f"   ✅ Test labels: "
        f"{test_labels.shape}"
    )


    # Clear GPU memory
    if device == "cuda":

        del chemberta_model

        torch.cuda.empty_cache()


    # =====================================================
    # Load Trained BiLSTM Model
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "🏗️  Loading BiLSTM model"
    )

    print("=" * 70)

    try:

        bilstm_model = EmbeddingBiLSTM(
            embedding_dim=embedding_dim,
            hidden_dim=256,
            lstm_hidden_1=128,
            lstm_hidden_2=256,
            lstm_hidden_3=512,
            lstm_dropout=0.3,
            fc_dropout=0.3
        ).to(device)

        # Load model parameters
        bilstm_model.load_state_dict(
            torch.load(
                model_path,
                map_location=device
            )
        )

        bilstm_model.eval()

        print(
            "✅ Model parameters loaded successfully"
        )

        # Calculate total number of parameters
        total_params = sum(
            p.numel()
            for p in bilstm_model.parameters()
        )

        print(
            f"📊 Total model parameters: "
            f"{total_params:,}"
        )

    except Exception as e:

        print(
            f"❌ Failed to load BiLSTM model: "
            f"{e}"
        )

        sys.exit(1)


    # =====================================================
    # Test Set Prediction
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "🚀 Starting prediction"
    )

    print("=" * 70)

    test_X = torch.FloatTensor(
        test_embeddings
    ).to(device)

    with torch.no_grad():

        predictions = (
            bilstm_model(test_X)
            .cpu()
            .numpy()
        )

    print(
        f"✅ Prediction completed for "
        f"{len(predictions)} samples"
    )


    # =====================================================
    # Calculate Evaluation Metrics
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "📊 Test Set Evaluation Results"
    )

    print("=" * 70)

    metrics = calculate_metrics(
        predictions,
        test_labels
    )

    print(
        "\n🎯 Independent test set performance metrics:"
    )

    print(
        f"   MSE:             "
        f"{metrics['mse']:.6f}"
    )

    print(
        f"   RMSE:            "
        f"{metrics['rmse']:.6f}"
    )

    print(
        f"   MAE:             "
        f"{metrics['mae']:.6f}"
    )

    print(
        f"   Pearson:         "
        f"{metrics['pearson']:.6f}"
    )

    print(
        f"   Pearson p-value: "
        f"{metrics['p_value']:.6e}"
    )

    print(
        f"   R²:              "
        f"{metrics['r2']:.6f}"
    )


    # =====================================================
    # Save Prediction Results
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "💾 Saving prediction results"
    )

    print("=" * 70)


    # Create result DataFrame
    results_df = (
        test_df_clean.copy()
    )

    results_df['Predicted'] = (
        predictions
    )

    results_df['Actual'] = (
        test_labels
    )

    results_df['Error'] = (
        predictions - test_labels
    )

    results_df['Abs_Error'] = (
        np.abs(
            predictions - test_labels
        )
    )

    results_df['Squared_Error'] = (
        predictions - test_labels
    ) ** 2


    # Save detailed prediction results
    output_csv = (
        'test_predictions.csv'
    )

    results_df.to_csv(
        output_csv,
        index=False
    )

    print(
        f"✅ Prediction results saved to: "
        f"{output_csv}"
    )


    # Save evaluation metrics
    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_csv = (
        'test_metrics.csv'
    )

    metrics_df.to_csv(
        metrics_csv,
        index=False
    )

    print(
        f"✅ Evaluation metrics saved to: "
        f"{metrics_csv}"
    )


    # =====================================================
    # Statistical Analysis
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "📈 Prediction Statistical Analysis"
    )

    print("=" * 70)


    print(
        "\nActual value statistics:"
    )

    print(
        f"   Minimum: "
        f"{test_labels.min():.4f}"
    )

    print(
        f"   Maximum: "
        f"{test_labels.max():.4f}"
    )

    print(
        f"   Mean:    "
        f"{test_labels.mean():.4f}"
    )

    print(
        f"   Std:     "
        f"{test_labels.std():.4f}"
    )


    print(
        "\nPredicted value statistics:"
    )

    print(
        f"   Minimum: "
        f"{predictions.min():.4f}"
    )

    print(
        f"   Maximum: "
        f"{predictions.max():.4f}"
    )

    print(
        f"   Mean:    "
        f"{predictions.mean():.4f}"
    )

    print(
        f"   Std:     "
        f"{predictions.std():.4f}"
    )


    print(
        "\nError statistics:"
    )

    errors = (
        predictions - test_labels
    )

    print(
        f"   Mean Error (ME):          "
        f"{errors.mean():.6f}"
    )

    print(
        f"   Mean Absolute Error:      "
        f"{metrics['mae']:.6f}"
    )

    print(
        f"   Root Mean Squared Error:  "
        f"{metrics['rmse']:.6f}"
    )

    print(
        f"   Error Standard Deviation: "
        f"{errors.std():.6f}"
    )


    # Calculate sample proportions within different error thresholds
    print(
        "\nError distribution:"
    )

    for threshold in [
        0.5,
        1.0,
        1.5,
        2.0
    ]:

        within_threshold = np.sum(
            np.abs(errors) <= threshold
        )

        percentage = (
            within_threshold
            / len(errors)
        ) * 100

        print(
            f"   Error <= {threshold}: "
            f"{within_threshold}/"
            f"{len(errors)} "
            f"({percentage:.2f}%)"
        )


    print(
        "\n" + "=" * 70
    )

    print(
        "🎉 Test completed successfully!"
    )

    print("=" * 70)