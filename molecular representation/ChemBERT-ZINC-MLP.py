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
        print("\nPlease run the following command to install the required dependency:")
        print("   pip install transformers")
        sys.exit(1)


# =====================================================
# Two-layer fully connected network definition
# =====================================================
class EmbeddingFCN(nn.Module):
    """Two-layer fully connected network for embedding vectors with dropout=0.3."""

    def __init__(
            self,
            embedding_dim=768,  # Default ChemBERTa-ZINC embedding dimension
            hidden_dim_1=512,  # First hidden layer dimension
            hidden_dim_2=256,  # Second hidden layer dimension
            fc_dropout=0.3
    ):
        super(EmbeddingFCN, self).__init__()

        # First fully connected layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.dropout1 = nn.Dropout(fc_dropout)

        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        self.dropout2 = nn.Dropout(fc_dropout)

        # Output layer
        self.fc_out = nn.Linear(hidden_dim_2, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, embedding_dim)

        # First fully connected layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Second fully connected layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Output layer
        x = self.fc_out(x)

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
def train_model(model, train_embeddings, train_labels, val_embeddings, val_labels,
                device, epochs=100, batch_size=32, learning_rate=0.001, seed=42):
    """Train the fully connected neural network model."""

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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

        # Mini-batch training
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + epoch)
        indices = torch.randperm(len(train_X), generator=generator)

        for i in range(0, len(train_X), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_X = train_X[batch_indices]
            batch_y = train_y[batch_indices]

            # Forward propagation
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward propagation
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation mode
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X)
            val_predictions = val_outputs.cpu().numpy()
            val_actuals = val_y.cpu().numpy()

            # Calculate all validation metrics
            val_metrics = calculate_metrics(val_predictions, val_actuals)

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['mse'])
        new_lr = optimizer.param_groups[0]['lr']

        # Calculate training metrics
        train_mse = np.mean(train_losses)

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
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        # Print results for each epoch
        print(f"\nEpoch [{epoch + 1:3d}/{epochs}]{is_best}")
        print(f"  Train MSE:   {train_mse:.6f}")
        print(f"  Val MSE:     {val_metrics['mse']:.6f}")
        print(f"  Val RMSE:    {val_metrics['rmse']:.6f}")
        print(f"  Val MAE:     {val_metrics['mae']:.6f}")
        print(f"  Val Pearson: {val_metrics['pearson']:.6f}")
        print(f"  Val R²:      {val_metrics['r2']:.6f}")
        print(f"  {lr_info}")

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            print(f"   Validation MSE did not improve for {early_stop_patience} consecutive epochs")
            break

    print("\n" + "=" * 70)
    print("Training completed")
    print(f"Best validation metrics (Epoch {best_epoch}):")
    print(f"   MSE:     {best_metrics['mse']:.6f}")
    print(f"   RMSE:    {best_metrics['rmse']:.6f}")
    print(f"   MAE:     {best_metrics['mae']:.6f}")
    print(f"   Pearson: {best_metrics['pearson']:.6f}")
    print(f"   R²:      {best_metrics['r2']:.6f}")
    print("=" * 70)

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))

    return best_metrics


# =====================================================
# Environment configuration
# =====================================================
def setup_environment():
    """Configure environment variables and Hugging Face mirror."""
    print("Configuring download environment...")
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    print("Hugging Face mirror enabled")


# =====================================================
# Main program
# =====================================================
if __name__ == "__main__":

    # =====================================================
    # Set random seed first
    # =====================================================
    RANDOM_SEED = 42
    set_seed(RANDOM_SEED)

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
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    batch_size_embedding = 16
    batch_size_training = 32
    epochs = 100
    learning_rate = 0.001

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("ChemBERTa-ZINC + Two-Layer Fully Connected Network for Molecular Property Prediction")
    print("=" * 70)
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch version: {torch.__version__}")

    print(f"Random seed: {RANDOM_SEED}")
    print(f"Pretrained model: {model_name}")
    print("=" * 70)

    # =====================================================
    # Read data
    # =====================================================
    print("\nReading datasets...")

    # Read training set
    try:
        train_df = pd.read_csv(train_csv_path, encoding='utf-8')
        print(f"Training set: {len(train_df)} rows")
    except Exception as e:
        print(f"Failed to read training set: {e}")
        sys.exit(1)

    # Read validation set
    try:
        val_df = pd.read_csv(val_csv_path, encoding='utf-8')
        print(f"Validation set: {len(val_df)} rows")
    except Exception as e:
        print(f"Failed to read validation set: {e}")
        sys.exit(1)

    # Check whether required columns exist
    for df, name in [(train_df, "training set"), (val_df, "validation set")]:
        if smiles_column not in df.columns:
            print(f"Column '{smiles_column}' does not exist in the {name}")
            print(f"Available columns: {', '.join(df.columns.tolist())}")
            sys.exit(1)

        if target_column not in df.columns:
            print(f"Column '{target_column}' does not exist in the {name}")
            print(f"Available columns: {', '.join(df.columns.tolist())}")
            sys.exit(1)

    # =====================================================
    # Load ChemBERTa-ZINC model
    # =====================================================
    print("\n" + "=" * 70)
    print("Loading ChemBERTa-ZINC model")
    print("=" * 70)

    setup_environment()

    # Lazy import transformers
    print("Loading transformers library...")
    TokenizerClass, ModelClass = lazy_import_transformers()
    print("Transformers imported successfully")

    # Load model
    print(f"\nPreparing to load model: {model_name}")
    print("The model needs to be downloaded on first use. Please wait...")

    try:
        print("   Downloading tokenizer...")
        tokenizer = TokenizerClass.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        print("   Downloading model...")
        chemberta_model = ModelClass.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        chemberta_model = chemberta_model.to(device)
        chemberta_model.eval()

        embedding_dim = chemberta_model.config.hidden_size

        print("Model loaded successfully")
        print(f"Embedding dimension: {embedding_dim}")

    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nPlease check the network connection or try downloading the model manually")
        sys.exit(1)

    # =====================================================
    # Extract embedding vectors
    # =====================================================
    print("\n" + "=" * 70)
    print("Extracting embedding vectors")
    print("=" * 70)

    print("\nProcessing training set...")
    train_embeddings, train_df_clean = extract_embeddings(
        train_df,
        smiles_column,
        tokenizer,
        chemberta_model,
        device,
        batch_size_embedding
    )

    train_labels = train_df_clean[target_column].values

    print(f"   Training embeddings: {train_embeddings.shape}")

    print("\nProcessing validation set...")
    val_embeddings, val_df_clean = extract_embeddings(
        val_df,
        smiles_column,
        tokenizer,
        chemberta_model,
        device,
        batch_size_embedding
    )

    val_labels = val_df_clean[target_column].values

    print(f"   Validation embeddings: {val_embeddings.shape}")

    # Clear memory
    if device == "cuda":
        del chemberta_model
        torch.cuda.empty_cache()

    # =====================================================
    # Build and train two-layer fully connected network
    # =====================================================
    print("\n" + "=" * 70)
    print("Building two-layer fully connected network")
    print("=" * 70)

    fcn_model = EmbeddingFCN(
        embedding_dim=embedding_dim,  # Automatically match ChemBERTa-ZINC dimension
        hidden_dim_1=512,  # First hidden layer
        hidden_dim_2=256,  # Second hidden layer
        fc_dropout=0.3
    ).to(device)

    # Print model structure information
    total_params = sum(
        p.numel()
        for p in fcn_model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in fcn_model.parameters()
        if p.requires_grad
    )

    print("\nModel parameters:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Train model
    best_metrics = train_model(
        fcn_model,
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
    print("\n" + "=" * 70)
    print("Final evaluation")
    print("=" * 70)

    fcn_model.eval()

    with torch.no_grad():
        val_X = torch.FloatTensor(val_embeddings).to(device)

        predictions = fcn_model(val_X).cpu().numpy()

        actual = val_labels

        final_metrics = calculate_metrics(
            predictions,
            actual
        )

        print("Validation performance:")
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

    results_df.to_csv(
        'predictions.csv',
        index=False
    )

    print("\nPrediction results saved to: predictions.csv")
    print("Best model saved to: best_model.pth")
    print(f"Random seed used: {RANDOM_SEED}")

    print("\n" + "=" * 70)
    print("All tasks completed")
    print("=" * 70)

