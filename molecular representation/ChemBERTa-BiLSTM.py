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
# Set random seed
# =====================================================
def set_seed(seed=42):
    """Set all random seeds to ensure reproducibility."""
    print(f"Setting random seed: {seed}")

    # Python random seed
    random.seed(seed)

    # NumPy random seed
    np.random.seed(seed)

    # PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # Ensure deterministic CUDA behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set the global Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    print("Random seed setup completed")


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

        # Treat the embedding vector as a sequence of length 1
        # Shape: (batch, seq_len=1, input_size)
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
# Training function
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
    """Train the BiLSTM model."""

    # Set training-specific random seed
    torch.manual_seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed(seed)

    # Convert arrays to tensors
    train_X = torch.FloatTensor(train_embeddings).to(device)
    train_y = torch.FloatTensor(train_labels).to(device)
    val_X = torch.FloatTensor(val_embeddings).to(device)
    val_y = torch.FloatTensor(val_labels).to(device)

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

    print("\n" + "=" * 70)
    print("Starting training")
    print("=" * 70)

    for epoch in range(epochs):

        # Training mode
        model.train()
        train_losses = []

        # Mini-batch training with deterministic shuffling
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + epoch)

        indices = torch.randperm(
            len(train_X),
            generator=generator
        )

        for i in range(0, len(train_X), batch_size):

            batch_indices = indices[i:i + batch_size]

            batch_X = train_X[batch_indices]
            batch_y = train_y[batch_indices]

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

            val_outputs = model(val_X)

            val_predictions = val_outputs.cpu().numpy()

            val_actuals = val_y.cpu().numpy()

            # Calculate all validation metrics
            val_metrics = calculate_metrics(
                val_predictions,
                val_actuals
            )

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']

        scheduler.step(
            val_metrics['mse']
        )

        new_lr = optimizer.param_groups[0]['lr']

        # Calculate training metrics
        train_mse = np.mean(
            train_losses
        )

        # Print results for each epoch
        lr_info = f"LR: {new_lr:.6f}"

        if new_lr != old_lr:
            lr_info += f" (reduced from {old_lr:.6f})"

        # Check whether this is the best model
        is_best = ""

        if val_metrics['mse'] < best_val_mse:

            is_best = " New best!"

            best_val_mse = val_metrics['mse']

            best_metrics = val_metrics.copy()

            best_epoch = epoch + 1

            patience_counter = 0

            torch.save(
                model.state_dict(),
                'best_model.pth'
            )

        else:

            patience_counter += 1

        # Print results for each epoch
        print(
            f"\nEpoch [{epoch + 1:3d}/{epochs}]{is_best}"
        )

        print(
            f"  Train MSE:   {train_mse:.6f}"
        )

        print(
            f"  Val MSE:     {val_metrics['mse']:.6f}"
        )

        print(
            f"  Val RMSE:    {val_metrics['rmse']:.6f}"
        )

        print(
            f"  Val MAE:     {val_metrics['mae']:.6f}"
        )

        print(
            f"  Val Pearson: {val_metrics['pearson']:.6f}"
        )

        print(
            f"  Val R²:      {val_metrics['r2']:.6f}"
        )

        print(
            f"  {lr_info}"
        )

        # Early stopping
        if patience_counter >= early_stop_patience:

            print(
                f"\nEarly stopping at epoch {epoch + 1}"
            )

            print(
                f"   Validation MSE did not improve for "
                f"{early_stop_patience} consecutive epochs"
            )

            break

    print("\n" + "=" * 70)

    print(
        "Training completed"
    )

    print(
        f"Best validation metrics (Epoch {best_epoch}):"
    )

    print(
        f"   MSE:     {best_metrics['mse']:.6f}"
    )

    print(
        f"   RMSE:    {best_metrics['rmse']:.6f}"
    )

    print(
        f"   MAE:     {best_metrics['mae']:.6f}"
    )

    print(
        f"   Pearson: {best_metrics['pearson']:.6f}"
    )

    print(
        f"   R²:      {best_metrics['r2']:.6f}"
    )

    print("=" * 70)

    # Load the best model
    model.load_state_dict(
        torch.load('best_model.pth')
    )

    return best_metrics


# =====================================================
# Environment configuration
# =====================================================
def setup_environment():
    """Configure the environment and download mirror."""

    print(
        "Configuring download environment..."
    )

    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    print(
        "Hugging Face mirror enabled"
    )


# =====================================================
# Main program
# =====================================================
if __name__ == "__main__":

    # =====================================================
    # Set random seed first
    # =====================================================
    RANDOM_SEED = 42

    set_seed(
        RANDOM_SEED
    )

    # =====================================================
    # Parameter settings
    # =====================================================

    # File paths
    train_csv_path = r"D:\pycharm\gutingle\pythonProject2\新\train_data.csv"

    val_csv_path = r"D:\pycharm\gutingle\pythonProject2\新\validation_data.csv"

    # Column names
    smiles_column = "Smiles"

    target_column = "pchembl"

    # Model parameters
    model_name = "DeepChem/ChemBERTa-77M-MLM"

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
        "ChemBERTa + BiLSTM Molecular Property Prediction"
    )

    print("=" * 70)

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"PyTorch version: {torch.__version__}"
        )

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print("=" * 70)

    # =====================================================
    # Read data
    # =====================================================
    print(
        "\nReading datasets..."
    )

    # Read training set
    try:

        train_df = pd.read_csv(
            train_csv_path,
            encoding='utf-8'
        )

        print(
            f"Training set: {len(train_df)} rows"
        )

    except Exception as e:

        print(
            f"Failed to read training set: {e}"
        )

        sys.exit(1)

    # Read validation set
    try:

        val_df = pd.read_csv(
            val_csv_path,
            encoding='utf-8'
        )

        print(
            f"Validation set: {len(val_df)} rows"
        )

    except Exception as e:

        print(
            f"Failed to read validation set: {e}"
        )

        sys.exit(1)

    # Check whether required columns exist
    for df, name in [
        (train_df, "training set"),
        (val_df, "validation set")
    ]:

        if smiles_column not in df.columns:

            print(
                f"Column '{smiles_column}' does not exist in the {name}"
            )

            print(
                f"Available columns: {', '.join(df.columns.tolist())}"
            )

            sys.exit(1)

        if target_column not in df.columns:

            print(
                f"Column '{target_column}' does not exist in the {name}"
            )

            print(
                f"Available columns: {', '.join(df.columns.tolist())}"
            )

            sys.exit(1)

    # =====================================================
    # Load ChemBERTa model
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "Loading ChemBERTa model"
    )

    print("=" * 70)

    setup_environment()

    # Lazy import transformers
    print(
        "Loading transformers library..."
    )

    TokenizerClass, ModelClass = lazy_import_transformers()

    print(
        "Transformers imported successfully"
    )

    # Load model
    print(
        f"\nPreparing to load model: {model_name}"
    )

    print(
        "Approximately 300 MB must be downloaded on first use. Please wait..."
    )

    try:

        print(
            "   Downloading tokenizer..."
        )

        tokenizer = TokenizerClass.from_pretrained(
            model_name,
            trust_remote_code=False
        )

        print(
            "   Downloading model..."
        )

        chemberta_model = ModelClass.from_pretrained(
            model_name,
            trust_remote_code=False
        )

        chemberta_model = chemberta_model.to(
            device
        )

        chemberta_model.eval()

        embedding_dim = chemberta_model.config.hidden_size

        print(
            "Model loaded successfully"
        )

        print(
            f"Embedding dimension: {embedding_dim}"
        )

    except Exception as e:

        print(
            f"Failed to load model: {e}"
        )

        print(
            "\nPlease check the network connection or try downloading the model manually"
        )

        sys.exit(1)

    # =====================================================
    # Extract embedding vectors
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "Extracting embedding vectors"
    )

    print("=" * 70)

    print(
        "\nProcessing training set..."
    )

    train_embeddings, train_df_clean = extract_embeddings(
        train_df,
        smiles_column,
        tokenizer,
        chemberta_model,
        device,
        batch_size_embedding
    )

    train_labels = train_df_clean[
        target_column
    ].values

    print(
        f"   Training embeddings: {train_embeddings.shape}"
    )

    print(
        "\nProcessing validation set..."
    )

    val_embeddings, val_df_clean = extract_embeddings(
        val_df,
        smiles_column,
        tokenizer,
        chemberta_model,
        device,
        batch_size_embedding
    )

    val_labels = val_df_clean[
        target_column
    ].values

    print(
        f"   Validation embeddings: {val_embeddings.shape}"
    )

    # Clear memory
    if device == "cuda":

        del chemberta_model

        torch.cuda.empty_cache()

    # =====================================================
    # Build and train BiLSTM model
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "Building BiLSTM model"
    )

    print("=" * 70)

    bilstm_model = EmbeddingBiLSTM(
        embedding_dim=embedding_dim,
        hidden_dim=256,
        lstm_hidden_1=128,
        lstm_hidden_2=256,
        lstm_hidden_3=512,
        lstm_dropout=0.3,
        fc_dropout=0.3
    ).to(device)

    # Print model parameter information
    total_params = sum(
        p.numel()
        for p in bilstm_model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in bilstm_model.parameters()
        if p.requires_grad
    )

    print(
        "\nModel parameters:"
    )

    print(
        f"   Total parameters: {total_params:,}"
    )

    print(
        f"   Trainable parameters: {trainable_params:,}"
    )

    # Train model
    best_metrics = train_model(
        bilstm_model,
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
    # Final evaluation
    # =====================================================
    print(
        "\n" + "=" * 70
    )

    print(
        "Final evaluation"
    )

    print("=" * 70)

    bilstm_model.eval()

    with torch.no_grad():

        val_X = torch.FloatTensor(
            val_embeddings
        ).to(device)

        predictions = bilstm_model(
            val_X
        ).cpu().numpy()

        actual = val_labels

        final_metrics = calculate_metrics(
            predictions,
            actual
        )

        print(
            "Validation performance:"
        )

        print(
            f"   MSE:     {final_metrics['mse']:.6f}"
        )

        print(
            f"   RMSE:    {final_metrics['rmse']:.6f}"
        )

        print(
            f"   MAE:     {final_metrics['mae']:.6f}"
        )

        print(
            f"   Pearson: {final_metrics['pearson']:.6f}"
        )

        print(
            f"   R²:      {final_metrics['r2']:.6f}"
        )

    # Save prediction results
    results_df = val_df_clean.copy()

    results_df[
        'Predicted'
    ] = predictions

    results_df[
        'Actual'
    ] = actual

    results_df[
        'Error'
    ] = predictions - actual

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
        "\nPrediction results saved to: predictions.csv"
    )

    print(
        "Best model saved to: best_model.pth"
    )

    print(
        f"Random seed used: {RANDOM_SEED}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "All tasks completed"
    )

    print("=" * 70)

