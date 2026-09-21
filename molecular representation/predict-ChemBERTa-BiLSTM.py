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
    """Lazy import transformers to avoid triggering torchvision issues."""
    try:
        from transformers import AutoTokenizer, AutoModel
        return AutoTokenizer, AutoModel
    except Exception as e:
        print(f"Import failed: {e}")
        print("\nPlease run the following commands to fix the dependencies:")
        print("   pip uninstall torch torchvision -y")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)


# =====================================================
# BiLSTM model definition
# Identical to the training code
# =====================================================
class EmbeddingBiLSTM(nn.Module):
    """Three-layer bidirectional LSTM for embedding vectors with dropout=0.3 in LSTM and FC layers."""

    def __init__(
        self,
        embedding_dim=384,
        hidden_dim=256,
        lstm_hidden_1=128,
        lstm_hidden_2=256,
        lstm_hidden_3=512,
        lstm_dropout=0.3,
        fc_dropout=0.3
    ):
        super(EmbeddingBiLSTM, self).__init__()

        # Layer 1: input=embedding_dim -> hidden=128
        # Bidirectional output channels = 256
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
        # Bidirectional output channels = 512
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
        # Bidirectional output channels = 1024
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
        # x: (batch, embedding_dim)
        x = x.unsqueeze(1)

        # Layer 1 (bidirectional)
        x, _ = self.lstm1(x)
        x = x.squeeze(1)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.unsqueeze(1)

        # Layer 2 (bidirectional)
        x, _ = self.lstm2(x)
        x = x.squeeze(1)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.unsqueeze(1)

        # Layer 3 (bidirectional)
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
# Embedding extraction function
# =====================================================
def extract_embeddings(df, smiles_column, tokenizer, model, device, batch_size=16):
    """Extract embedding vectors from a DataFrame."""

    # Data cleaning
    original_count = len(df)
    df = df.dropna(subset=[smiles_column])
    df = df[df[smiles_column].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)

    removed_count = original_count - len(df)
    if removed_count > 0:
        print(f"   Removed {removed_count} invalid samples")

    smiles_list = df[smiles_column].astype(str).tolist()
    print(f"   Number of valid SMILES: {len(smiles_list)}")

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
            print(f"\n   Batch processing failed: {e}")
            embedding_dim = model.config.hidden_size
            zero_embeddings = np.zeros((len(batch), embedding_dim))
            all_embeddings.append(zero_embeddings)

    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings, df


# =====================================================
# Calculate evaluation metrics
# =====================================================
def calculate_metrics(predictions, actuals):
    """Calculate regression evaluation metrics."""

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
# Environment configuration
# =====================================================
def setup_environment():
    """Configure environment variables and download mirror."""
    print("Configuring download environment...")
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print("Hugging Face mirror enabled")


# =====================================================
# Main program - Test set evaluation
# =====================================================
if __name__ == "__main__":

    # =====================================================
    # Parameter settings
    # =====================================================

    # File paths
    test_csv_path = r"D:\pycharm\gutingle\pythonProject2\新\test_data.csv"
    model_path = "best_model.pth"

    # Column names
    smiles_column = "Smiles"
    target_column = "pchembl"

    # Model parameters
    # Must be consistent with training
    model_name = "DeepChem/ChemBERTa-77M-MLM"
    batch_size_embedding = 16

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("ChemBERTa + BiLSTM Independent Test Set Evaluation")
    print("=" * 70)
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch version: {torch.__version__}")

    print("=" * 70)

    # =====================================================
    # Read test set
    # =====================================================
    print("\nReading test set...")

    try:
        test_df = pd.read_csv(
            test_csv_path,
            encoding='utf-8'
        )

        print(f"Test set: {len(test_df)} rows")

    except Exception as e:
        print(f"Failed to read test set: {e}")
        sys.exit(1)

    # Check whether required columns exist
    if smiles_column not in test_df.columns:
        print(f"Column '{smiles_column}' does not exist in the test set")
        print(f"Available columns: {', '.join(test_df.columns.tolist())}")
        sys.exit(1)

    if target_column not in test_df.columns:
        print(f"Column '{target_column}' does not exist in the test set")
        print(f"Available columns: {', '.join(test_df.columns.tolist())}")
        sys.exit(1)

    # =====================================================
    # Load ChemBERTa model
    # =====================================================
    print("\n" + "=" * 70)
    print("Loading ChemBERTa model")
    print("=" * 70)

    setup_environment()

    # Lazy import transformers
    print("Loading transformers library...")

    TokenizerClass, ModelClass = lazy_import_transformers()

    print("Transformers imported successfully")

    # Load model
    print(f"\nPreparing to load model: {model_name}")

    try:
        print("   Loading tokenizer...")

        tokenizer = TokenizerClass.from_pretrained(
            model_name,
            trust_remote_code=False
        )

        print("   Loading model...")

        chemberta_model = ModelClass.from_pretrained(
            model_name,
            trust_remote_code=False
        )

        chemberta_model = chemberta_model.to(device)
        chemberta_model.eval()

        embedding_dim = chemberta_model.config.hidden_size

        print("Model loaded successfully")
        print(f"Embedding dimension: {embedding_dim}")

    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # =====================================================
    # Extract test set embeddings
    # =====================================================
    print("\n" + "=" * 70)
    print("Extracting test set embeddings")
    print("=" * 70)

    test_embeddings, test_df_clean = extract_embeddings(
        test_df,
        smiles_column,
        tokenizer,
        chemberta_model,
        device,
        batch_size_embedding
    )

    test_labels = test_df_clean[target_column].values

    print(f"   Test set embeddings: {test_embeddings.shape}")

    # Clear memory
    if device == "cuda":
        del chemberta_model
        torch.cuda.empty_cache()

    # =====================================================
    # Load trained BiLSTM model
    # =====================================================
    print("\n" + "=" * 70)
    print("Loading trained BiLSTM model")
    print("=" * 70)

    # Check whether the model file exists
    if not os.path.exists(model_path):
        print(f"Model file does not exist: {model_path}")
        print("Please make sure the model has been trained and saved")
        sys.exit(1)

    # Create model instance
    # Parameters must be consistent with training
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
    try:
        bilstm_model.load_state_dict(
            torch.load(
                model_path,
                map_location=device
            )
        )

        bilstm_model.eval()

        print(f"Successfully loaded model: {model_path}")

        total_params = sum(
            p.numel()
            for p in bilstm_model.parameters()
        )

        print(f"Model parameters: {total_params:,}")

    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # =====================================================
    # Predict on the test set
    # =====================================================
    print("\n" + "=" * 70)
    print("Test set prediction")
    print("=" * 70)

    with torch.no_grad():
        test_X = torch.FloatTensor(
            test_embeddings
        ).to(device)

        predictions = bilstm_model(
            test_X
        ).cpu().numpy()

        actual = test_labels

    # =====================================================
    # Calculate evaluation metrics
    # =====================================================
    print("\n" + "=" * 70)
    print("Test set evaluation results")
    print("=" * 70)

    test_metrics = calculate_metrics(
        predictions,
        actual
    )

    print("\nIndependent test set performance:")
    print(f"   MSE:                 {test_metrics['mse']:.6f}")
    print(f"   RMSE:                {test_metrics['rmse']:.6f}")
    print(f"   MAE:                 {test_metrics['mae']:.6f}")
    print(f"   Pearson correlation: {test_metrics['pearson']:.6f}")
    print(f"   R²:                  {test_metrics['r2']:.6f}")

    # =====================================================
    # Save prediction results
    # =====================================================
    print("\n" + "=" * 70)
    print("Saving prediction results")
    print("=" * 70)

    # Create result DataFrame
    results_df = test_df_clean.copy()

    results_df['Predicted'] = predictions
    results_df['Actual'] = actual
    results_df['Error'] = predictions - actual
    results_df['Abs_Error'] = np.abs(
        predictions - actual
    )

    # Save detailed results
    output_file = 'test_predictions.csv'

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Detailed prediction results saved to: {output_file}"
    )

    # Save evaluation metrics
    metrics_file = 'test_metrics.txt'

    with open(
        metrics_file,
        'w',
        encoding='utf-8'
    ) as f:

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "Independent Test Set Evaluation Metrics\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"MSE:                 {test_metrics['mse']:.6f}\n"
        )

        f.write(
            f"RMSE:                {test_metrics['rmse']:.6f}\n"
        )

        f.write(
            f"MAE:                 {test_metrics['mae']:.6f}\n"
        )

        f.write(
            f"Pearson correlation: {test_metrics['pearson']:.6f}\n"
        )

        f.write(
            f"R²:                  {test_metrics['r2']:.6f}\n"
        )

        f.write(
            "\n" + "=" * 70 + "\n"
        )

        f.write(
            f"Number of test samples: {len(predictions)}\n"
        )

        f.write(
            f"Model file: {model_path}\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

    print(
        f"Evaluation metrics saved to: {metrics_file}"
    )

    # =====================================================
    # Display prediction examples
    # =====================================================
    print("\n" + "=" * 70)
    print("Prediction examples (first 10)")
    print("=" * 70)

    sample_df = results_df.head(10)[
        [
            'Smiles',
            'Actual',
            'Predicted',
            'Error',
            'Abs_Error'
        ]
    ]

    print(
        sample_df.to_string(
            index=False
        )
    )

    # Display error statistics
    print("\n" + "=" * 70)
    print("Error statistics")
    print("=" * 70)

    print(
        f"Maximum positive error: {results_df['Error'].max():.6f}"
    )

    print(
        f"Maximum negative error: {results_df['Error'].min():.6f}"
    )

    print(
        f"Mean error:             {results_df['Error'].mean():.6f}"
    )

    print(
        f"Error standard deviation: {results_df['Error'].std():.6f}"
    )

    print("\n" + "=" * 70)
    print("Test completed")
    print("=" * 70)

