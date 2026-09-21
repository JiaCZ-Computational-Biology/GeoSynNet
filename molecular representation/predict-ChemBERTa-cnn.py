import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import sys
import warnings
from scipy.stats import pearsonr
import random

warnings.filterwarnings('ignore')


def lazy_import_transformers():
    try:
        from transformers import AutoTokenizer, AutoModel
        return AutoTokenizer, AutoModel
    except Exception as e:
        print(f"Failed to import transformers: {e}")
        print("\nPlease run the following commands to fix the dependencies:")
        print("   pip uninstall torch torchvision -y")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)


class EmbeddingCNN(nn.Module):

    def __init__(self, embedding_dim=384, hidden_dim=256):
        super(EmbeddingCNN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=5,
            padding=2
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            padding=2
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=5,
            padding=2
        )
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(256, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze(-1)


def extract_embeddings(
    df,
    smiles_column,
    tokenizer,
    model,
    device,
    batch_size=16
):

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

            zero_embeddings = np.zeros(
                (len(batch), embedding_dim)
            )

            all_embeddings.append(zero_embeddings)

    all_embeddings = np.vstack(all_embeddings)

    return all_embeddings, df


def calculate_metrics(predictions, actual):

    mse = np.mean(
        (predictions - actual) ** 2
    )

    rmse = np.sqrt(mse)

    mae = np.mean(
        np.abs(predictions - actual)
    )

    ss_res = np.sum(
        (actual - predictions) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    r2 = 1 - (ss_res / ss_tot)

    pearson_corr, pearson_pvalue = pearsonr(
        predictions,
        actual
    )

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Pearson_r': pearson_corr,
        'Pearson_p': pearson_pvalue
    }


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def setup_environment():

    print("Configuring download environment...")

    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    print("Hugging Face mirror enabled")


print("=" * 70)
print("ChemBERTa + CNN Independent Test Set Evaluation")
print("=" * 70)

SEED = 42

set_seed(SEED)

test_csv_path = r"test_data.csv"

model_path = "best_model.pth"

smiles_column = "Smiles"

target_column = "pchembl"

model_name = "DeepChem/ChemBERTa-77M-MLM"

batch_size_embedding = 16

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")

print("=" * 70)


if not os.path.exists(model_path):

    print(f"Model file does not exist: {model_path}")

    print(
        "Please make sure that the training script "
        "successfully saved the model."
    )

    sys.exit(1)


print("\nLoading test dataset...")

try:

    test_df = pd.read_csv(
        test_csv_path,
        encoding='utf-8'
    )

    print(
        f"Test dataset: {len(test_df)} rows"
    )

except Exception as e:

    print(
        f"Failed to load test dataset: {e}"
    )

    sys.exit(1)


if smiles_column not in test_df.columns:

    print(
        f"Column '{smiles_column}' "
        f"does not exist in the test dataset."
    )

    print(
        f"Available columns: "
        f"{', '.join(test_df.columns.tolist())}"
    )

    sys.exit(1)


if target_column not in test_df.columns:

    print(
        f"Column '{target_column}' "
        f"does not exist in the test dataset."
    )

    print(
        f"Available columns: "
        f"{', '.join(test_df.columns.tolist())}"
    )

    sys.exit(1)


print("\n" + "=" * 70)

print("Loading ChemBERTa Model")

print("=" * 70)

setup_environment()

print("Loading transformers library...")

TokenizerClass, ModelClass = lazy_import_transformers()

print("Transformers imported successfully")

print(f"\nLoading model: {model_name}")

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

    print("Model loaded successfully!")

    print(
        f"Embedding dimension: {embedding_dim}"
    )

except Exception as e:

    print(
        f"Failed to load model: {e}"
    )

    print(
        "\nPlease check your network connection "
        "or download the model manually."
    )

    sys.exit(1)


print("\n" + "=" * 70)

print("Extracting Test Set Embeddings")

print("=" * 70)


test_embeddings, test_df_clean = extract_embeddings(
    test_df,
    smiles_column,
    tokenizer,
    chemberta_model,
    device,
    batch_size_embedding
)


test_labels = test_df_clean[
    target_column
].values


print(
    f"Test embeddings: "
    f"{test_embeddings.shape}"
)


if device == "cuda":

    del chemberta_model

    torch.cuda.empty_cache()


print("\n" + "=" * 70)

print("Loading Trained CNN Model")

print("=" * 70)


try:

    cnn_model = EmbeddingCNN(
        embedding_dim=embedding_dim,
        hidden_dim=256
    ).to(device)

    cnn_model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    cnn_model.eval()

    total_params = sum(
        p.numel()
        for p in cnn_model.parameters()
    )

    print("Model loaded successfully")

    print(
        f"Total model parameters: "
        f"{total_params:,}"
    )

except Exception as e:

    print(
        f"Failed to load model: {e}"
    )

    sys.exit(1)


print("\n" + "=" * 70)

print("Predicting on Test Set")

print("=" * 70)


with torch.no_grad():

    test_X = torch.FloatTensor(
        test_embeddings
    ).to(device)

    predictions = cnn_model(
        test_X
    ).cpu().numpy()


print(
    f"Prediction completed for "
    f"{len(predictions)} samples"
)


print("\n" + "=" * 70)

print("Test Set Evaluation Metrics")

print("=" * 70)


metrics = calculate_metrics(
    predictions,
    test_labels
)


print("\nTest Set Performance:")

print(
    f"   {'Metric':<25} "
    f"{'Value':>15}"
)

print(
    f"   {'-' * 40}"
)

print(
    f"   {'MSE':<25} "
    f"{metrics['MSE']:>15.6f}"
)

print(
    f"   {'RMSE':<25} "
    f"{metrics['RMSE']:>15.6f}"
)

print(
    f"   {'MAE':<25} "
    f"{metrics['MAE']:>15.6f}"
)

print(
    f"   {'R2':<25} "
    f"{metrics['R2']:>15.6f}"
)

print(
    f"   {'Pearson r':<25} "
    f"{metrics['Pearson_r']:>15.6f}"
)

print(
    f"   {'Pearson p-value':<25} "
    f"{metrics['Pearson_p']:>15.6e}"
)


print("\n" + "=" * 70)

print("Saving Prediction Results")

print("=" * 70)


results_df = test_df_clean.copy()

results_df['Predicted'] = predictions

results_df['Actual'] = test_labels

results_df['Error'] = (
    predictions - test_labels
)

results_df['Abs_Error'] = np.abs(
    predictions - test_labels
)

results_df['Squared_Error'] = (
    predictions - test_labels
) ** 2


output_file = 'test_predictions.csv'

results_df.to_csv(
    output_file,
    index=False
)

print(
    f"Detailed prediction results "
    f"saved to: {output_file}"
)


metrics_df = pd.DataFrame(
    [metrics]
)

metrics_file = 'test_metrics.csv'

metrics_df.to_csv(
    metrics_file,
    index=False
)

print(
    f"Evaluation metrics saved to: "
    f"{metrics_file}"
)


print("\n" + "=" * 70)

print("Prediction Statistics")

print("=" * 70)


print("\nActual Value Statistics:")

print(
    f"   Minimum: "
    f"{test_labels.min():.4f}"
)

print(
    f"   Maximum: "
    f"{test_labels.max():.4f}"
)

print(
    f"   Mean: "
    f"{test_labels.mean():.4f}"
)

print(
    f"   Standard deviation: "
    f"{test_labels.std():.4f}"
)


print("\nPredicted Value Statistics:")

print(
    f"   Minimum: "
    f"{predictions.min():.4f}"
)

print(
    f"   Maximum: "
    f"{predictions.max():.4f}"
)

print(
    f"   Mean: "
    f"{predictions.mean():.4f}"
)

print(
    f"   Standard deviation: "
    f"{predictions.std():.4f}"
)


print("\nError Statistics:")

errors = (
    predictions - test_labels
)

print(
    f"   Mean error: "
    f"{errors.mean():.4f}"
)

print(
    f"   Error standard deviation: "
    f"{errors.std():.4f}"
)

print(
    f"   Maximum positive error: "
    f"{errors.max():.4f}"
)

print(
    f"   Maximum negative error: "
    f"{errors.min():.4f}"
)


print("\n" + "=" * 70)

print("Test Set Evaluation Completed!")

print("=" * 70)

print("\nGenerated files:")

print(
    f"   1. {output_file} "
    f"- Detailed prediction results"
)

print(
    f"   2. {metrics_file} "
    f"- Evaluation metrics summary"
)

print(
    f"\nRandom seed: {SEED}"
)

print("=" * 70)