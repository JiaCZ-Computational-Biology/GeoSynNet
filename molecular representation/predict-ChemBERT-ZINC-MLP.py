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
        print("\nPlease run the following command to install the required dependency:")
        print("   pip install transformers")
        sys.exit(1)


# =====================================================
# Two-layer fully connected network definition
# Consistent with the training code
# =====================================================
class EmbeddingFCN(nn.Module):
    """Two-layer fully connected network for embedding vectors with dropout=0.3."""

    def __init__(
            self,
            embedding_dim=768,
            hidden_dim_1=512,
            hidden_dim_2=256,
            fc_dropout=0.3
    ):
        super(EmbeddingFCN, self).__init__()

        self.fc1 = nn.Linear(embedding_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.dropout1 = nn.Dropout(fc_dropout)

        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        self.dropout2 = nn.Dropout(fc_dropout)

        self.fc_out = nn.Linear(hidden_dim_2, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc_out(x)

        return x.squeeze(-1)


# =====================================================
# Embedding extraction function
# =====================================================
def extract_embeddings(df, smiles_column, tokenizer, model, device, batch_size=16):
    """Extract embedding vectors from a DataFrame."""
    df = df.dropna(subset=[smiles_column])
    df = df[df[smiles_column].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)

    smiles_list = df[smiles_column].astype(str).tolist()
    all_embeddings = []

    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Extracting embeddings"):
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
            embedding_dim = model.config.hidden_size
            zero_embeddings = np.zeros((len(batch), embedding_dim))
            all_embeddings.append(zero_embeddings)

    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings, df


# =====================================================
# Environment configuration
# =====================================================
def setup_environment():
    """Configure environment variables and Hugging Face mirror."""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'


# =====================================================
# Main program - Independent test set evaluation
# =====================================================
if __name__ == "__main__":

    # =====================================================
    # Parameter settings
    # =====================================================
    test_csv_path = r"test_data.csv"
    model_path = "best_model.pth"

    smiles_column = "Smiles"
    target_column = "pchembl"

    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    batch_size_embedding = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Independent test set evaluation")
    print(f"Device: {device}")

    # =====================================================
    # Check model file
    # =====================================================
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    # =====================================================
    # Read test set
    # =====================================================
    print("\nReading test set...")
    test_df = pd.read_csv(test_csv_path, encoding='utf-8')
    print(f"Test set: {len(test_df)} rows")

    # =====================================================
    # Load ChemBERTa model
    # =====================================================
    print("\nLoading ChemBERTa model...")
    setup_environment()

    TokenizerClass, ModelClass = lazy_import_transformers()

    tokenizer = TokenizerClass.from_pretrained(model_name, trust_remote_code=True)
    chemberta_model = ModelClass.from_pretrained(model_name, trust_remote_code=True)
    chemberta_model = chemberta_model.to(device)
    chemberta_model.eval()

    embedding_dim = chemberta_model.config.hidden_size
    print(f"Model loaded successfully, embedding dimension: {embedding_dim}")

    # =====================================================
    # Extract test set embeddings
    # =====================================================
    print("\nExtracting test set embeddings...")
    test_embeddings, test_df_clean = extract_embeddings(
        test_df, smiles_column, tokenizer, chemberta_model, device, batch_size_embedding
    )
    test_labels = test_df_clean[target_column].values

    if device == "cuda":
        del chemberta_model
        torch.cuda.empty_cache()

    # =====================================================
    # Load trained model
    # =====================================================
    print("\nLoading trained model...")
    fcn_model = EmbeddingFCN(
        embedding_dim=embedding_dim,
        hidden_dim_1=512,
        hidden_dim_2=256,
        fc_dropout=0.3
    ).to(device)

    fcn_model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully")

    # =====================================================
    # Prediction
    # =====================================================
    print("\nRunning predictions...")
    fcn_model.eval()

    with torch.no_grad():
        test_X = torch.FloatTensor(test_embeddings).to(device)
        predictions = fcn_model(test_X).cpu().numpy()
        actual = test_labels

    # =====================================================
    # Calculate metrics
    # =====================================================
    mse = np.mean((predictions - actual) ** 2)
    mae = np.mean(np.abs(predictions - actual))
    pearson_corr, _ = pearsonr(predictions, actual)

    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # =====================================================
    # Output results
    # =====================================================
    print("\n" + "=" * 50)
    print("Test set evaluation results:")
    print("=" * 50)
    print(f"MSE:                 {mse:.6f}")
    print(f"MAE:                 {mae:.6f}")
    print(f"Pearson correlation: {pearson_corr:.6f}")
    print(f"R²:                  {r2:.6f}")
    print("=" * 50)

