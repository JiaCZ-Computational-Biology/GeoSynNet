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
# Set Random Seed
# =====================================================
def set_seed(seed=42):
    """Set all random seeds to ensure reproducibility."""

    print(f"🌱 Setting random seed: {seed}")

    # Python random seed
    random.seed(seed)

    # NumPy random seed
    np.random.seed(seed)

    # PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic CUDA behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set global Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    print("✅ Random seed configuration completed")


# =====================================================
# Lazy Import of Transformers
# =====================================================
def lazy_import_transformers():
    """Lazily import transformers to avoid torchvision-related issues."""

    try:
        from transformers import AutoTokenizer, AutoModel
        return AutoTokenizer, AutoModel

    except Exception as e:
        print(f"❌ Import failed: {e}")
        print("\n💡 Please run the following command to install dependencies:")
        print("   pip install transformers")
        sys.exit(1)


# =====================================================
# CNN Model Definition
# =====================================================
class EmbeddingCNN(nn.Module):
    """Three-layer CNN for processing embedding vectors with dropout=0.3."""

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

        # Convolution layer 1: input_channels=1 -> output_channels=128
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=cnn_out_channels_1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.bn1 = nn.BatchNorm1d(cnn_out_channels_1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(cnn_dropout)

        # Convolution layer 2: input_channels=128 -> output_channels=256
        self.conv2 = nn.Conv1d(
            in_channels=cnn_out_channels_1,
            out_channels=cnn_out_channels_2,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.bn2 = nn.BatchNorm1d(cnn_out_channels_2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(cnn_dropout)

        # Convolution layer 3: input_channels=256 -> output_channels=512
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
        # Input shape: (batch, embedding_dim)
        # Convert to (batch, 1, embedding_dim) for 1D convolution
        x = x.unsqueeze(1)

        # Convolution layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Convolution layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Convolution layer 3
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
    """Train the CNN model."""

    # Set training-specific random seed
    torch.manual_seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed(seed)

    # Convert arrays to tensors
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

    # Define loss function and optimizer
    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    best_val_mse = float('inf')
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = 50
    best_metrics = None

    print("\n" + "=" * 70)
    print("🚀 Starting training")
    print("=" * 70)

    for epoch in range(epochs):

        # Training mode
        model.train()

        train_losses = []

        # Mini-batch training
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + epoch)

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

            # Forward propagation
            optimizer.zero_grad()

            outputs = model(batch_X)

            loss = criterion(
                outputs,
                batch_y
            )

            # Backward propagation
            loss.backward()

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        # Validation mode
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

            # Calculate validation metrics
            val_metrics = calculate_metrics(
                val_predictions,
                val_actuals
            )

        # Learning rate scheduling
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

        # Calculate training metric
        train_mse = np.mean(
            train_losses
        )

        # Learning rate information
        lr_info = (
            f"LR: {new_lr:.6f}"
        )

        if new_lr != old_lr:

            lr_info += (
                f" ⬇️ "
                f"(reduced from {old_lr:.6f})"
            )

        # Check whether this is the best model
        is_best = ""

        if val_metrics['mse'] < best_val_mse:

            is_best = " ⭐ New best!"

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

        # Print results for every epoch
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
            f"  Val R²:      "
            f"{val_metrics['r2']:.6f}"
        )

        print(
            f"  {lr_info}"
        )

        # Early stopping
        if patience_counter >= early_stop_patience:

            print(
                f"\n⚠️  Early stopping at "
                f"epoch {epoch + 1}"
            )

            print(
                f"   Validation MSE has not improved "
                f"for {early_stop_patience} consecutive epochs"
            )

            break

    print("\n" + "=" * 70)

    print(
        "✅ Training completed!"
    )

    print(
        f"🏆 Best validation metrics "
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
        f"   R²:      "
        f"{best_metrics['r2']:.6f}"
    )

    print("=" * 70)

    # Load the best model
    model.load_state_dict(
        torch.load(
            'best_model.pth'
        )
    )

    return best_metrics


# =====================================================
# Environment Configuration
# =====================================================
def setup_environment():
    """Configure the download environment."""

    print(
        "🌐 Configuring download environment..."
    )

    os.environ[
        'HF_ENDPOINT'
    ] = 'https://hf-mirror.com'

    os.environ[
        'HF_HUB_ENABLE_HF_TRANSFER'
    ] = '0'

    print(
        "✅ Mirror acceleration enabled"
    )


# =====================================================
# Main Program
# =====================================================
if __name__ == "__main__":

    # =====================================================
    # Set Random Seed
    # =====================================================
    RANDOM_SEED = 42

    set_seed(
        RANDOM_SEED
    )

    # =====================================================
    # Parameter Settings
    # =====================================================

    # File paths
    train_csv_path = (
        r"train_data.csv"
    )

    val_csv_path = (
        r"validation_data.csv"
    )

    # Column names
    smiles_column = "Smiles"

    target_column = "pchembl"

    # Model parameters
    model_name = (
        "seyonec/ChemBERTa-zinc-base-v1"
    )

    batch_size_embedding = 16

    batch_size_training = 32

    epochs = 100

    learning_rate = 0.001

    # Device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)

    print(
        "🧪 ChemBERTa-ZINC + CNN "
        "Molecular Property Prediction"
    )

    print("=" * 70)

    print(
        f"🔧 Device: "
        f"{device}"
    )

    if torch.cuda.is_available():

        print(
            f"🎮 GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"📦 PyTorch version: "
            f"{torch.__version__}"
        )

    print(
        f"🌱 Random seed: "
        f"{RANDOM_SEED}"
    )

    print(
        f"📦 Pretrained model: "
        f"{model_name}"
    )

    print("=" * 70)


    # =====================================================
    # Load Datasets
    # =====================================================
    print(
        "\n📂 Loading datasets..."
    )

    # Load training dataset
    try:

        train_df = pd.read_csv(
            train_csv_path,
            encoding='utf-8'
        )

        print(
            f"✅ Training set: "
            f"{len(train_df)} rows"
        )

    except Exception as e:

        print(
            f"❌ Failed to load training set: "
            f"{e}"
        )

        sys.exit(1)

    # Load validation dataset
    try:

        val_df = pd.read_csv(
            val_csv_path,
            encoding='utf-8'
        )

        print(
            f"✅ Validation set: "
            f"{len(val_df)} rows"
        )

    except Exception as e:

        print(
            f"❌ Failed to load validation set: "
            f"{e}"
        )

        sys.exit(1)

    # Check required columns
    for df, name in [
        (train_df, "training set"),
        (val_df, "validation set")
    ]:

        if smiles_column not in df.columns:

            print(
                f"❌ Column '{smiles_column}' "
                f"does not exist in the {name}"
            )

            print(
                f"💡 Available columns: "
                f"{', '.join(df.columns.tolist())}"
            )

            sys.exit(1)

        if target_column not in df.columns:

            print(
                f"❌ Column '{target_column}' "
                f"does not exist in the {name}"
            )

            print(
                f"💡 Available columns: "
                f"{', '.join(df.columns.tolist())}"
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

    # Load model
    print(
        f"\n📦 Preparing to load model: "
        f"{model_name}"
    )

    print(
        "⏳ The model will be downloaded "
        "if it is not available locally..."
    )

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
            chemberta_model
            .config
            .hidden_size
        )

        print(
            "✅ Model loaded successfully!"
        )

        print(
            f"📐 Embedding dimension: "
            f"{embedding_dim}"
        )

    except Exception as e:

        print(
            f"❌ Failed to load model: "
            f"{e}"
        )

        print(
            "\n💡 Please check your network connection "
            "or try downloading the model manually"
        )

        sys.exit(1)


    # =====================================================
    # Extract Embedding Vectors
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "🔄 Extracting embedding vectors"
    )

    print("=" * 70)

    print(
        "\n📊 Processing training set..."
    )

    train_embeddings, train_df_clean = (
        extract_embeddings(
            train_df,
            smiles_column,
            tokenizer,
            chemberta_model,
            device,
            batch_size_embedding
        )
    )

    train_labels = (
        train_df_clean[
            target_column
        ].values
    )

    print(
        f"   ✅ Training embeddings: "
        f"{train_embeddings.shape}"
    )

    print(
        "\n📊 Processing validation set..."
    )

    val_embeddings, val_df_clean = (
        extract_embeddings(
            val_df,
            smiles_column,
            tokenizer,
            chemberta_model,
            device,
            batch_size_embedding
        )
    )

    val_labels = (
        val_df_clean[
            target_column
        ].values
    )

    print(
        f"   ✅ Validation embeddings: "
        f"{val_embeddings.shape}"
    )

    # Clear GPU memory
    if device == "cuda":

        del chemberta_model

        torch.cuda.empty_cache()


    # =====================================================
    # Build and Train CNN Model
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "🏗️  Building CNN model "
        "(three convolutional layers)"
    )

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

    # Print model parameter statistics
    total_params = sum(
        p.numel()
        for p in cnn_model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in cnn_model.parameters()
        if p.requires_grad
    )

    print(
        "\n📊 Model parameters:"
    )

    print(
        f"   Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"   Trainable parameters: "
        f"{trainable_params:,}"
    )

    # Train model
    best_metrics = train_model(
        cnn_model,
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
        "📊 Final Evaluation"
    )

    print("=" * 70)

    cnn_model.eval()

    with torch.no_grad():

        val_X = torch.FloatTensor(
            val_embeddings
        ).to(device)

        predictions = (
            cnn_model(val_X)
            .cpu()
            .numpy()
        )

        actual = val_labels

        final_metrics = calculate_metrics(
            predictions,
            actual
        )

        print(
            "🎯 Validation set performance:"
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
            f"   R²:      "
            f"{final_metrics['r2']:.6f}"
        )


    # Save prediction results
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
        predictions - actual
    )

    results_df[
        'Abs_Error'
    ] = np.abs(
        predictions - actual
    )

    results_df.to_csv(
        'predictions.csv',
        index=False
    )

    print(
        "\n💾 Prediction results saved to: "
        "predictions.csv"
    )

    print(
        "💾 Best model saved to: "
        "best_model.pth"
    )

    print(
        f"🌱 Random seed used: "
        f"{RANDOM_SEED}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "🎉 All tasks completed successfully!"
    )

    print("=" * 70)