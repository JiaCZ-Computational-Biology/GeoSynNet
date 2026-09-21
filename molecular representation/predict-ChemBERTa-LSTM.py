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


class EmbeddingLSTM(nn.Module):

    def __init__(self, embedding_dim=384, hidden_dim=256):
        super(EmbeddingLSTM, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        self.bn1 = nn.BatchNorm1d(128)

        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        self.bn2 = nn.BatchNorm1d(256)

        self.lstm3 = nn.LSTM(
            input_size=256,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        self.bn3 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)

        x, (h1, c1) = self.lstm1(x)
        x = x.squeeze(1)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.unsqueeze(1)

        x, (h2, c2) = self.lstm2(x)
        x = x.squeeze(1)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.unsqueeze(1)

        x, (h3, c3) = self.lstm3(x)
        x = x.squeeze(1)
        x = self.bn3(x)
        x = self.relu(x)

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

    df = df[
        df[smiles_column]
        .astype(str)
        .str.strip() != ""
    ]

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


def calculate_metrics(predictions, actuals):

    mse = np.mean(
        (predictions - actuals) ** 2
    )

    rmse = np.sqrt(mse)

    mae = np.mean(
        np.abs(predictions - actuals)
    )

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

    try:
        pearson_corr, p_value = pearsonr(
            predictions,
            actuals
        )

    except Exception:
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


def setup_environment():

    print("Configuring download environment...")

    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    print("Hugging Face mirror enabled")


if __name__ == "__main__":

    test_csv_path = (
        r"test_data.csv"
    )

    model_path = "best_model.pth"

    smiles_column = "Smiles"

    target_column = "pchembl"

    chemberta_model_name = "DeepChem/ChemBERTa-77M-MLM"

    batch_size_embedding = 16

    embedding_dim = 384

    hidden_dim = 256

    device = "cuda" if torch.cuda.is_available() else "cpu"


    print("=" * 70)

    print("ChemBERTa + LSTM Independent Test Set Evaluation")

    print("=" * 70)

    print(f"Device: {device}")


    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"PyTorch version: "
            f"{torch.__version__}"
        )


    print("=" * 70)


    print("\nLoading test dataset...")


    try:

        test_df = pd.read_csv(
            test_csv_path,
            encoding='utf-8'
        )

        print(
            f"Test dataset: "
            f"{len(test_df)} rows"
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


    print(
        f"\nPreparing to load model: "
        f"{chemberta_model_name}"
    )


    try:

        print("   Loading tokenizer...")


        tokenizer = TokenizerClass.from_pretrained(
            chemberta_model_name,
            trust_remote_code=False
        )


        print("   Loading model...")


        chemberta_model = ModelClass.from_pretrained(
            chemberta_model_name,
            trust_remote_code=False
        )


        chemberta_model = chemberta_model.to(device)


        chemberta_model.eval()


        embedding_dim = chemberta_model.config.hidden_size


        print("Model loaded successfully!")


        print(
            f"Embedding dimension: "
            f"{embedding_dim}"
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
        f"   Test embeddings: "
        f"{test_embeddings.shape}"
    )


    print(
        f"   Number of test labels: "
        f"{len(test_labels)}"
    )


    if device == "cuda":

        del chemberta_model

        torch.cuda.empty_cache()


    print("\n" + "=" * 70)

    print("Loading Trained LSTM Model")

    print("=" * 70)


    try:

        lstm_model = EmbeddingLSTM(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        ).to(device)


        lstm_model.load_state_dict(
            torch.load(
                model_path,
                map_location=device
            )
        )


        lstm_model.eval()


        total_params = sum(
            p.numel()
            for p in lstm_model.parameters()
        )


        print(
            "Model loaded successfully!"
        )


        print(
            f"Total model parameters: "
            f"{total_params:,}"
        )


    except Exception as e:

        print(
            f"Failed to load model: {e}"
        )


        print(
            f"Please make sure that "
            f"the model file "
            f"'{model_path}' exists."
        )


        sys.exit(1)


    print("\n" + "=" * 70)

    print("Predicting on Test Set")

    print("=" * 70)


    test_X = torch.FloatTensor(
        test_embeddings
    ).to(device)


    with torch.no_grad():

        predictions = (
            lstm_model(
                test_X
            )
            .cpu()
            .numpy()
        )


    print(
        "Prediction completed!"
    )


    print(
        f"   Prediction range: "
        f"[{predictions.min():.4f}, "
        f"{predictions.max():.4f}]"
    )


    print(
        f"   Actual value range: "
        f"[{test_labels.min():.4f}, "
        f"{test_labels.max():.4f}]"
    )


    print("\n" + "=" * 70)

    print("Test Set Evaluation Metrics")

    print("=" * 70)


    metrics = calculate_metrics(
        predictions,
        test_labels
    )


    print(
        "\nIndependent Test Set Performance:"
    )


    print(
        f"   MSE:       "
        f"{metrics['mse']:.6f}"
    )


    print(
        f"   RMSE:      "
        f"{metrics['rmse']:.6f}"
    )


    print(
        f"   MAE:       "
        f"{metrics['mae']:.6f}"
    )


    print(
        f"   R2:        "
        f"{metrics['r2']:.6f}"
    )


    print(
        f"   Pearson r: "
        f"{metrics['pearson']:.6f}"
    )


    print(
        f"   P-value:   "
        f"{metrics['p_value']:.6e}"
    )


    print("\n" + "=" * 70)

    print("Saving Prediction Results")

    print("=" * 70)


    results_df = test_df_clean.copy()


    results_df[
        'Predicted'
    ] = predictions


    results_df[
        'Actual'
    ] = test_labels


    results_df[
        'Error'
    ] = (
        predictions - test_labels
    )


    results_df[
        'Abs_Error'
    ] = np.abs(
        predictions - test_labels
    )


    results_df[
        'Squared_Error'
    ] = (
        predictions - test_labels
    ) ** 2


    results_df.to_csv(
        'test_predictions.csv',
        index=False,
        encoding='utf-8'
    )


    print(
        "Detailed prediction results "
        "saved to: test_predictions.csv"
    )


    metrics_df = pd.DataFrame(
        [metrics]
    )


    metrics_df.to_csv(
        'test_metrics.csv',
        index=False,
        encoding='utf-8'
    )


    print(
        "Evaluation metrics saved to: "
        "test_metrics.csv"
    )


    print("\n" + "=" * 70)

    print("Error Statistics")

    print("=" * 70)


    errors = (
        predictions - test_labels
    )


    abs_errors = np.abs(
        errors
    )


    print(
        "\nError Distribution:"
    )


    print(
        f"   Mean error: "
        f"{np.mean(errors):.6f}"
    )


    print(
        f"   Error standard deviation: "
        f"{np.std(errors):.6f}"
    )


    print(
        f"   Maximum positive error: "
        f"{np.max(errors):.6f}"
    )


    print(
        f"   Maximum negative error: "
        f"{np.min(errors):.6f}"
    )


    print(
        f"   Median absolute error: "
        f"{np.median(abs_errors):.6f}"
    )


    print(
        "\nError Range Distribution:"
    )


    for threshold in [
        0.1,
        0.2,
        0.5,
        1.0
    ]:

        count = (
            abs_errors <= threshold
        ).sum()


        percentage = (
            count /
            len(abs_errors) *
            100
        )


        print(
            f"   Absolute error <= "
            f"{threshold}: "
            f"{percentage:.2f}% "
            f"({count}/{len(abs_errors)})"
        )


    print("\n" + "=" * 70)

    print("Test Set Evaluation Completed!")

    print("=" * 70)