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
# Lazy Import of Transformers
# =====================================================
def lazy_import_transformers():
    """Import transformers lazily to avoid torchvision-related issues."""
    try:
        from transformers import AutoTokenizer, AutoModel
        return AutoTokenizer, AutoModel

    except Exception as e:
        print(f"Import failed: {e}")
        print("\nPlease run the following commands to fix dependencies:")
        print("pip uninstall torch torchvision -y")
        print(
            "pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cpu"
        )
        sys.exit(1)


# =====================================================
# Two-Layer MLP Model
# =====================================================
class EmbeddingMLP(nn.Module):
    """Two-layer MLP for processing embedding vectors."""

    def __init__(
            self,
            embedding_dim=384,
            hidden_dim_1=512,
            hidden_dim_2=256,
            dropout=0.3
    ):
        super(EmbeddingMLP, self).__init__()

        self.fc1 = nn.Linear(
            embedding_dim,
            hidden_dim_1
        )

        self.bn1 = nn.BatchNorm1d(
            hidden_dim_1
        )

        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.fc2 = nn.Linear(
            hidden_dim_1,
            hidden_dim_2
        )

        self.bn2 = nn.BatchNorm1d(
            hidden_dim_2
        )

        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(
            dropout
        )

        self.fc_out = nn.Linear(
            hidden_dim_2,
            1
        )

    def forward(self, x):
        # x: (batch, embedding_dim)

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
# Embedding Extraction
# =====================================================
def extract_embeddings(
        df,
        smiles_column,
        tokenizer,
        model,
        device,
        batch_size=16
):
    """Extract embeddings from a DataFrame."""

    original_count = len(df)

    df = df.dropna(
        subset=[smiles_column]
    )

    df = df[
        df[smiles_column]
        .astype(str)
        .str.strip() != ""
    ]

    df = df.reset_index(
        drop=True
    )

    removed_count = (
        original_count
        - len(df)
    )

    if removed_count > 0:
        print(
            f"Removed {removed_count} invalid samples."
        )

    smiles_list = (
        df[smiles_column]
        .astype(str)
        .tolist()
    )

    print(
        f"Valid SMILES count: "
        f"{len(smiles_list)}"
    )

    all_embeddings = []

    for i in tqdm(
            range(
                0,
                len(smiles_list),
                batch_size
            ),
            desc="Extracting embeddings"
    ):
        batch = smiles_list[
            i:i + batch_size
        ]

        try:
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(
                    **inputs
                )

                cls_embeddings = (
                    outputs
                    .last_hidden_state[:, 0, :]
                )

                cls_embeddings = (
                    cls_embeddings
                    .cpu()
                    .numpy()
                )

                all_embeddings.append(
                    cls_embeddings
                )

        except Exception as e:
            print(
                f"\nBatch processing failed: {e}"
            )

            embedding_dim = (
                model.config.hidden_size
            )

            zero_embeddings = np.zeros(
                (
                    len(batch),
                    embedding_dim
                )
            )

            all_embeddings.append(
                zero_embeddings
            )

    all_embeddings = np.vstack(
        all_embeddings
    )

    return all_embeddings, df


# =====================================================
# Evaluation Metrics
# =====================================================
def calculate_metrics(
        predictions,
        actuals
):
    """Calculate regression evaluation metrics."""

    mse = np.mean(
        (predictions - actuals) ** 2
    )

    mae = np.mean(
        np.abs(
            predictions
            - actuals
        )
    )

    ss_res = np.sum(
        (
            actuals
            - predictions
        ) ** 2
    )

    ss_tot = np.sum(
        (
            actuals
            - np.mean(actuals)
        ) ** 2
    )

    r2 = (
        1 - (ss_res / ss_tot)
        if ss_tot != 0
        else 0
    )

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
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr,
        'p_value': p_value
    }


# =====================================================
# Environment Setup
# =====================================================
def setup_environment():
    """Configure the model download environment."""

    print(
        "Configuring download environment..."
    )

    os.environ[
        'HF_ENDPOINT'
    ] = 'https://hf-mirror.com'

    print(
        "Hugging Face mirror enabled."
    )


# =====================================================
# Main Program - Independent Test Set Evaluation
# =====================================================
if __name__ == "__main__":

    # =====================================================
    # Parameters
    # =====================================================
    test_csv_path = (
        r"test_data.csv"
    )

    model_path = (
        "best_model.pth"
    )

    smiles_column = "Smiles"

    target_column = "pchembl"

    chemberta_model_name = (
        "DeepChem/ChemBERTa-77M-MLM"
    )

    batch_size_embedding = 16

    embedding_dim = 384

    hidden_dim_1 = 512

    hidden_dim_2 = 256

    dropout = 0.3

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 70
    )

    print(
        "Independent Test Set Evaluation - "
        "ChemBERTa + Two-Layer MLP"
    )

    print(
        "=" * 70
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"PyTorch version: "
            f"{torch.__version__}"
        )

    print(
        "=" * 70
    )

    # =====================================================
    # Load Test Data
    # =====================================================
    print(
        "\nLoading test dataset..."
    )

    try:
        test_df = pd.read_csv(
            test_csv_path,
            encoding='utf-8'
        )

        print(
            f"Test set: "
            f"{len(test_df)} rows"
        )

    except Exception as e:
        print(
            f"Failed to load "
            f"test set: {e}"
        )

        sys.exit(1)

    if smiles_column not in test_df.columns:
        print(
            f"Column '{smiles_column}' "
            f"does not exist in the test set."
        )

        print(
            f"Available columns: "
            f"{', '.join(test_df.columns.tolist())}"
        )

        sys.exit(1)

    if target_column not in test_df.columns:
        print(
            f"Column '{target_column}' "
            f"does not exist in the test set."
        )

        print(
            f"Available columns: "
            f"{', '.join(test_df.columns.tolist())}"
        )

        sys.exit(1)

    # =====================================================
    # Load ChemBERTa Model
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "Loading ChemBERTa Model"
    )

    print(
        "=" * 70
    )

    setup_environment()

    print(
        "Loading transformers library..."
    )

    TokenizerClass, ModelClass = (
        lazy_import_transformers()
    )

    print(
        "Transformers imported successfully."
    )

    print(
        f"\nLoading model: "
        f"{chemberta_model_name}"
    )

    try:
        print(
            "Loading tokenizer..."
        )

        tokenizer = (
            TokenizerClass
            .from_pretrained(
                chemberta_model_name,
                trust_remote_code=False
            )
        )

        print(
            "Loading ChemBERTa model..."
        )

        chemberta_model = (
            ModelClass
            .from_pretrained(
                chemberta_model_name,
                trust_remote_code=False
            )
        )

        chemberta_model = (
            chemberta_model
            .to(device)
        )

        chemberta_model.eval()

        embedding_dim = (
            chemberta_model
            .config
            .hidden_size
        )

        print(
            "ChemBERTa model loaded successfully."
        )

        print(
            f"Embedding dimension: "
            f"{embedding_dim}"
        )

    except Exception as e:
        print(
            f"Failed to load model: {e}"
        )

        sys.exit(1)

    # =====================================================
    # Extract Test Set Embeddings
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "Extracting Test Set Embeddings"
    )

    print(
        "=" * 70
    )

    print(
        "\nProcessing test set..."
    )

    (
        test_embeddings,
        test_df_clean
    ) = extract_embeddings(
        test_df,
        smiles_column,
        tokenizer,
        chemberta_model,
        device,
        batch_size_embedding
    )

    test_labels = (
        test_df_clean[
            target_column
        ].values
    )

    print(
        f"Test embeddings: "
        f"{test_embeddings.shape}"
    )

    print(
        f"Test labels: "
        f"{len(test_labels)}"
    )

    if device == "cuda":
        del chemberta_model

        torch.cuda.empty_cache()

    # =====================================================
    # Load Trained MLP Model
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "Loading Trained MLP Model"
    )

    print(
        "=" * 70
    )

    if not os.path.exists(
        model_path
    ):
        print(
            f"Model file does not exist: "
            f"{model_path}"
        )

        print(
            f"Please make sure the trained "
            f"model is saved as "
            f"'{model_path}'."
        )

        sys.exit(1)

    mlp_model = EmbeddingMLP(
        embedding_dim=embedding_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        dropout=dropout
    ).to(device)

    try:
        mlp_model.load_state_dict(
            torch.load(
                model_path,
                map_location=device
            )
        )

        mlp_model.eval()

        print(
            f"Model loaded successfully: "
            f"{model_path}"
        )

        total_params = sum(
            p.numel()
            for p in mlp_model.parameters()
        )

        print(
            f"Model parameters: "
            f"{total_params:,}"
        )

    except Exception as e:
        print(
            f"Failed to load model: {e}"
        )

        sys.exit(1)

    # =====================================================
    # Test Set Prediction
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "Test Set Prediction"
    )

    print(
        "=" * 70
    )

    mlp_model.eval()

    with torch.no_grad():

        test_X = torch.FloatTensor(
            test_embeddings
        ).to(device)

        predictions = (
            mlp_model(
                test_X
            )
            .cpu()
            .numpy()
        )

        actual = test_labels

    print(
        "Prediction completed."
    )

    print(
        f"Number of predicted samples: "
        f"{len(predictions)}"
    )

    # =====================================================
    # Evaluation
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "Test Set Evaluation Results"
    )

    print(
        "=" * 70
    )

    test_metrics = calculate_metrics(
        predictions,
        actual
    )

    print(
        "\nTest set performance metrics:"
    )

    print(
        f"   {'Metric':<15} "
        f"{'Value':>12}"
    )

    print(
        f"   {'-' * 28}"
    )

    print(
        f"   {'MSE':<15} "
        f"{test_metrics['mse']:>12.6f}"
    )

    print(
        f"   {'MAE':<15} "
        f"{test_metrics['mae']:>12.6f}"
    )

    print(
        f"   {'Pearson':<15} "
        f"{test_metrics['pearson']:>12.6f}"
    )

    print(
        f"   {'R2':<15} "
        f"{test_metrics['r2']:>12.6f}"
    )

    # =====================================================
    # Save Prediction Results
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "Saving Prediction Results"
    )

    print(
        "=" * 70
    )

    results_df = (
        test_df_clean.copy()
    )

    results_df[
        'Predicted'
    ] = predictions

    results_df[
        'Actual'
    ] = actual

    results_df[
        'Error'
    ] = (
        predictions
        - actual
    )

    results_df[
        'Abs_Error'
    ] = np.abs(
        predictions
        - actual
    )

    results_df[
        'Squared_Error'
    ] = (
        predictions
        - actual
    ) ** 2

    output_file = (
        'test_predictions.csv'
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Detailed prediction results "
        f"saved to: {output_file}"
    )

    metrics_file = (
        'test_metrics.txt'
    )

    with open(
        metrics_file,
        'w',
        encoding='utf-8'
    ) as f:

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            "Test Set Evaluation Results\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Model path: "
            f"{model_path}\n"
        )

        f.write(
            f"Test set path: "
            f"{test_csv_path}\n"
        )

        f.write(
            f"Number of test samples: "
            f"{len(predictions)}\n\n"
        )

        f.write(
            f"{'Metric':<15} "
            f"{'Value':>12}\n"
        )

        f.write(
            f"{'-' * 28}\n"
        )

        f.write(
            f"{'MSE':<15} "
            f"{test_metrics['mse']:>12.6f}\n"
        )

        f.write(
            f"{'MAE':<15} "
            f"{test_metrics['mae']:>12.6f}\n"
        )

        f.write(
            f"{'Pearson':<15} "
            f"{test_metrics['pearson']:>12.6f}\n"
        )

        f.write(
            f"{'R2':<15} "
            f"{test_metrics['r2']:>12.6f}\n"
        )

    print(
        f"Evaluation metrics saved to: "
        f"{metrics_file}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "Test set evaluation completed."
    )

    print(
        "=" * 70
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"1. {output_file} - "
        f"Detailed prediction results"
    )

    print(
        f"2. {metrics_file} - "
        f"Evaluation metrics summary"
    )

    print(
        "=" * 70
    )