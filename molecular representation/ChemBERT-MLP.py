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

warnings.filterwarnings('ignore')


# =====================================================
# Random Seed Setup
# =====================================================
def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    print(f"Setting random seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

    print("Random seed setup completed.")


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
    """Two-layer MLP for processing molecular embeddings."""

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
    """Extract molecular embeddings from a DataFrame."""

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

    rmse = np.sqrt(
        mse
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
        pearson_corr, _ = pearsonr(
            predictions,
            actuals
        )

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
# Training Function
# =====================================================
def train_model(
        model,
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        device,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        seed=42
):
    """Train the MLP model."""

    torch.manual_seed(
        seed
    )

    if device == "cuda":
        torch.cuda.manual_seed(
            seed
        )

    train_X = torch.FloatTensor(
        train_embeddings
    ).to(device)

    train_y = torch.FloatTensor(
        train_labels
    ).to(device)

    val_X = torch.FloatTensor(
        val_embeddings
    ).to(device)

    val_y = torch.FloatTensor(
        val_labels
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
    )

    best_val_mse = float(
        'inf'
    )

    best_epoch = 0

    patience_counter = 0

    early_stop_patience = 50

    best_metrics = None

    print(
        "\n" + "=" * 70
    )

    print(
        "Training Started"
    )

    print(
        "=" * 70
    )

    for epoch in range(
        epochs
    ):
        model.train()

        train_losses = []

        generator = torch.Generator(
            device=device
        )

        generator.manual_seed(
            seed + epoch
        )

        indices = torch.randperm(
            len(train_X),
            generator=generator
        )

        for i in range(
                0,
                len(train_X),
                batch_size
        ):
            batch_indices = indices[
                i:i + batch_size
            ]

            batch_X = train_X[
                batch_indices
            ]

            batch_y = train_y[
                batch_indices
            ]

            optimizer.zero_grad()

            outputs = model(
                batch_X
            )

            loss = criterion(
                outputs,
                batch_y
            )

            loss.backward()

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        model.eval()

        with torch.no_grad():
            val_outputs = model(
                val_X
            )

            val_predictions = (
                val_outputs
                .cpu()
                .numpy()
            )

            val_actuals = (
                val_y
                .cpu()
                .numpy()
            )

            val_metrics = calculate_metrics(
                val_predictions,
                val_actuals
            )

        old_lr = (
            optimizer
            .param_groups[0]['lr']
        )

        scheduler.step(
            val_metrics['mse']
        )

        new_lr = (
            optimizer
            .param_groups[0]['lr']
        )

        train_mse = np.mean(
            train_losses
        )

        lr_info = (
            f"LR: {new_lr:.6f}"
        )

        if new_lr != old_lr:
            lr_info += (
                f" "
                f"(reduced from {old_lr:.6f})"
            )

        is_best = ""

        if (
            val_metrics['mse']
            < best_val_mse
        ):
            is_best = " NEW BEST"

            best_val_mse = (
                val_metrics['mse']
            )

            best_metrics = (
                val_metrics.copy()
            )

            best_epoch = (
                epoch + 1
            )

            patience_counter = 0

            torch.save(
                model.state_dict(),
                'best_model.pth'
            )

        else:
            patience_counter += 1

        print(
            f"\nEpoch "
            f"[{epoch + 1:3d}/{epochs}]"
            f"{is_best}"
        )

        print(
            f"  Train MSE:   "
            f"{train_mse:.6f}"
        )

        print(
            f"  Val MSE:     "
            f"{val_metrics['mse']:.6f}"
        )

        print(
            f"  Val RMSE:    "
            f"{val_metrics['rmse']:.6f}"
        )

        print(
            f"  Val MAE:     "
            f"{val_metrics['mae']:.6f}"
        )

        print(
            f"  Val Pearson: "
            f"{val_metrics['pearson']:.6f}"
        )

        print(
            f"  Val R2:      "
            f"{val_metrics['r2']:.6f}"
        )

        print(
            f"  {lr_info}"
        )

        if (
            patience_counter
            >= early_stop_patience
        ):
            print(
                f"\nEarly stopping at "
                f"epoch {epoch + 1}"
            )

            print(
                f"Validation MSE did not "
                f"improve for "
                f"{early_stop_patience} "
                f"consecutive epochs."
            )

            break

    print(
        "\n" + "=" * 70
    )

    print(
        "Training Completed"
    )

    print(
        f"Best validation metrics "
        f"(Epoch {best_epoch}):"
    )

    print(
        f"   MSE:     "
        f"{best_metrics['mse']:.6f}"
    )

    print(
        f"   RMSE:    "
        f"{best_metrics['rmse']:.6f}"
    )

    print(
        f"   MAE:     "
        f"{best_metrics['mae']:.6f}"
    )

    print(
        f"   Pearson: "
        f"{best_metrics['pearson']:.6f}"
    )

    print(
        f"   R2:      "
        f"{best_metrics['r2']:.6f}"
    )

    print(
        "=" * 70
    )

    model.load_state_dict(
        torch.load(
            'best_model.pth',
            map_location=device
        )
    )

    return best_metrics


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
# Main Program
# =====================================================
if __name__ == "__main__":

    # =====================================================
    # Random Seed
    # =====================================================
    RANDOM_SEED = 42

    set_seed(
        RANDOM_SEED
    )

    # =====================================================
    # Parameters
    # =====================================================
    train_csv_path = (
        r"train_data.csv"
    )

    val_csv_path = (
        r"validation_data.csv"
    )

    smiles_column = "Smiles"

    target_column = "pchembl"

    model_name = (
        "DeepChem/ChemBERTa-77M-MLM"
    )

    batch_size_embedding = 16

    batch_size_training = 32

    epochs = 1000

    learning_rate = 0.005

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 70
    )

    print(
        "ChemBERTa + Two-Layer MLP "
        "Molecular Property Prediction"
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
        f"Random seed: "
        f"{RANDOM_SEED}"
    )

    print(
        "=" * 70
    )

    # =====================================================
    # Load Data
    # =====================================================
    print(
        "\nLoading datasets..."
    )

    try:
        train_df = pd.read_csv(
            train_csv_path,
            encoding='utf-8'
        )

        print(
            f"Training set: "
            f"{len(train_df)} rows"
        )

    except Exception as e:
        print(
            f"Failed to load "
            f"training set: {e}"
        )

        sys.exit(1)

    try:
        val_df = pd.read_csv(
            val_csv_path,
            encoding='utf-8'
        )

        print(
            f"Validation set: "
            f"{len(val_df)} rows"
        )

    except Exception as e:
        print(
            f"Failed to load "
            f"validation set: {e}"
        )

        sys.exit(1)

    for df, name in [
        (
            train_df,
            "training set"
        ),
        (
            val_df,
            "validation set"
        )
    ]:
        if smiles_column not in df.columns:
            print(
                f"Column '{smiles_column}' "
                f"does not exist in the "
                f"{name}."
            )

            print(
                f"Available columns: "
                f"{', '.join(df.columns.tolist())}"
            )

            sys.exit(1)

        if target_column not in df.columns:
            print(
                f"Column '{target_column}' "
                f"does not exist in the "
                f"{name}."
            )

            print(
                f"Available columns: "
                f"{', '.join(df.columns.tolist())}"
            )

            sys.exit(1)

    # =====================================================
    # Load ChemBERTa
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
        f"\nPreparing model: "
        f"{model_name}"
    )

    print(
        "The model will be downloaded "
        "if it is not available locally."
    )

    try:
        print(
            "Loading tokenizer..."
        )

        tokenizer = (
            TokenizerClass
            .from_pretrained(
                model_name,
                trust_remote_code=False
            )
        )

        print(
            "Loading model..."
        )

        chemberta_model = (
            ModelClass
            .from_pretrained(
                model_name,
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
            "Model loaded successfully."
        )

        print(
            f"Embedding dimension: "
            f"{embedding_dim}"
        )

    except Exception as e:
        print(
            f"Failed to load model: {e}"
        )

        print(
            "\nPlease check the network "
            "connection or download the "
            "model manually."
        )

        sys.exit(1)

    # =====================================================
    # Extract Embeddings
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "Extracting Molecular Embeddings"
    )

    print(
        "=" * 70
    )

    print(
        "\nProcessing training set..."
    )

    (
        train_embeddings,
        train_df_clean
    ) = extract_embeddings(
        train_df,
        smiles_column,
        tokenizer,
        chemberta_model,
        device,
        batch_size_embedding
    )

    train_labels = (
        train_df_clean[
            target_column
        ].values
    )

    print(
        f"Training embeddings: "
        f"{train_embeddings.shape}"
    )

    print(
        "\nProcessing validation set..."
    )

    (
        val_embeddings,
        val_df_clean
    ) = extract_embeddings(
        val_df,
        smiles_column,
        tokenizer,
        chemberta_model,
        device,
        batch_size_embedding
    )

    val_labels = (
        val_df_clean[
            target_column
        ].values
    )

    print(
        f"Validation embeddings: "
        f"{val_embeddings.shape}"
    )

    if device == "cuda":
        del chemberta_model

        torch.cuda.empty_cache()

    # =====================================================
    # Build and Train the MLP
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "Building Two-Layer MLP"
    )

    print(
        "=" * 70
    )

    mlp_model = EmbeddingMLP(
        embedding_dim=embedding_dim,
        hidden_dim_1=512,
        hidden_dim_2=256,
        dropout=0.3
    ).to(device)

    total_params = sum(
        p.numel()
        for p in mlp_model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in mlp_model.parameters()
        if p.requires_grad
    )

    print(
        "\nModel parameters:"
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    print(
        "\nModel architecture:"
    )

    print(
        f"Input layer:  "
        f"{embedding_dim}"
    )

    print(
        "Hidden layer 1: "
        "512 (dropout=0.3)"
    )

    print(
        "Hidden layer 2: "
        "256 (dropout=0.3)"
    )

    print(
        "Output layer: 1"
    )

    best_metrics = train_model(
        mlp_model,
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        device,
        epochs=epochs,
        batch_size=batch_size_training,
        learning_rate=learning_rate,
        seed=RANDOM_SEED
    )

    # =====================================================
    # Final Evaluation
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "Final Evaluation"
    )

    print(
        "=" * 70
    )

    mlp_model.eval()

    with torch.no_grad():

        val_X = torch.FloatTensor(
            val_embeddings
        ).to(device)

        predictions = (
            mlp_model(
                val_X
            )
            .cpu()
            .numpy()
        )

        actual = (
            val_labels
        )

        final_metrics = calculate_metrics(
            predictions,
            actual
        )

        print(
            "Validation set performance:"
        )

        print(
            f"   MSE:     "
            f"{final_metrics['mse']:.6f}"
        )

        print(
            f"   RMSE:    "
            f"{final_metrics['rmse']:.6f}"
        )

        print(
            f"   MAE:     "
            f"{final_metrics['mae']:.6f}"
        )

        print(
            f"   Pearson: "
            f"{final_metrics['pearson']:.6f}"
        )

        print(
            f"   R2:      "
            f"{final_metrics['r2']:.6f}"
        )

    results_df = (
        val_df_clean.copy()
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

    results_df.to_csv(
        'predictions.csv',
        index=False
    )

    print(
        "\nPrediction results saved to: "
        "predictions.csv"
    )

    print(
        "Best model saved to: "
        "best_model.pth"
    )

    print(
        f"Random seed used: "
        f"{RANDOM_SEED}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "All tasks completed."
    )

    print(
        "=" * 70
    )