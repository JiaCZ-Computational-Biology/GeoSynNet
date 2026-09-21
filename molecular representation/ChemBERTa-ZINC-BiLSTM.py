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


def set_seed(seed=42):
    print(f"Setting random seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

    print("Random seed set successfully")


def lazy_import_transformers():
    try:
        from transformers import AutoTokenizer, AutoModel
        return AutoTokenizer, AutoModel

    except Exception as e:
        print(f"Failed to import transformers: {e}")
        print("\nPlease run the following command to install dependencies:")
        print("   pip install transformers")
        sys.exit(1)


class EmbeddingBiLSTM(nn.Module):

    def __init__(
        self,
        embedding_dim=768,
        hidden_dim=256,
        lstm_hidden_1=128,
        lstm_hidden_2=256,
        lstm_hidden_3=512,
        lstm_dropout=0.3,
        fc_dropout=0.3
    ):
        super(EmbeddingBiLSTM, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_1,
            num_layers=1,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True
        )

        self.bn1 = nn.BatchNorm1d(
            lstm_hidden_1 * 2
        )

        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden_1 * 2,
            hidden_size=lstm_hidden_2,
            num_layers=1,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True
        )

        self.bn2 = nn.BatchNorm1d(
            lstm_hidden_2 * 2
        )

        self.lstm3 = nn.LSTM(
            input_size=lstm_hidden_2 * 2,
            hidden_size=lstm_hidden_3,
            num_layers=1,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True
        )

        self.bn3 = nn.BatchNorm1d(
            lstm_hidden_3 * 2
        )

        fc_in = lstm_hidden_3 * 2

        self.fc1 = nn.Linear(
            fc_in,
            hidden_dim
        )

        self.dropout = nn.Dropout(
            fc_dropout
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            1
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.unsqueeze(1)

        x, _ = self.lstm1(x)

        x = x.squeeze(1)

        x = self.bn1(x)

        x = self.relu(x)

        x = x.unsqueeze(1)


        x, _ = self.lstm2(x)

        x = x.squeeze(1)

        x = self.bn2(x)

        x = self.relu(x)

        x = x.unsqueeze(1)


        x, _ = self.lstm3(x)

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
        original_count - len(df)
    )


    if removed_count > 0:

        print(
            f"   Removed {removed_count} invalid samples"
        )


    smiles_list = (
        df[smiles_column]
        .astype(str)
        .tolist()
    )


    print(
        f"   Number of valid SMILES: "
        f"{len(smiles_list)}"
    )


    all_embeddings = []


    for i in tqdm(
        range(
            0,
            len(smiles_list),
            batch_size
        ),
        desc="   Extracting embeddings"
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
                f"\n   Batch processing failed: {e}"
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


def calculate_metrics(
    predictions,
    actuals
):

    mse = np.mean(
        (
            predictions - actuals
        ) ** 2
    )


    rmse = np.sqrt(
        mse
    )


    mae = np.mean(
        np.abs(
            predictions - actuals
        )
    )


    ss_res = np.sum(
        (
            actuals - predictions
        ) ** 2
    )


    ss_tot = np.sum(
        (
            actuals - np.mean(actuals)
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

    except Exception:

        pearson_corr = 0.0


    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr
    }


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
        optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
    )


    best_val_mse = float('inf')

    best_epoch = 0

    patience_counter = 0

    early_stop_patience = 50


    print(
        "\n" + "=" * 70
    )

    print(
        "Starting Training"
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


            val_metrics = (
                calculate_metrics(
                    val_predictions,
                    val_actuals
                )
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
                f" (reduced from "
                f"{old_lr:.6f})"
            )


        is_best = ""


        if (
            val_metrics['mse']
            < best_val_mse
        ):

            is_best = " - New Best"

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
                f"\nEarly stopping "
                f"at epoch {epoch + 1}"
            )


            print(
                f"Validation MSE has not "
                f"improved for "
                f"{early_stop_patience} epochs."
            )


            break


    print(
        "\n" + "=" * 70
    )


    print(
        "Training Completed"
    )


    print(
        f"Best Validation Metrics "
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
            'best_model.pth'
        )
    )


    return best_metrics


def setup_environment():

    print(
        "Configuring download environment..."
    )


    os.environ[
        'HF_ENDPOINT'
    ] = 'https://hf-mirror.com'


    os.environ[
        'HF_HUB_ENABLE_HF_TRANSFER'
    ] = '0'


    print(
        "Hugging Face mirror enabled"
    )


if __name__ == "__main__":

    RANDOM_SEED = 42


    set_seed(
        RANDOM_SEED
    )


    train_csv_path = (
        r"train_data.csv"
    )


    val_csv_path = (
        r"validation_data.csv"
    )


    smiles_column = "Smiles"


    target_column = "pchembl"


    model_name = (
        "seyonec/ChemBERTa-zinc-base-v1"
    )


    batch_size_embedding = 16


    batch_size_training = 32


    epochs = 100


    learning_rate = 0.001


    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "=" * 70
    )


    print(
        "ChemBERTa-ZINC + BiLSTM "
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
        f"Pretrained model: "
        f"{model_name}"
    )


    print(
        "=" * 70
    )


    print(
        "\nLoading datasets..."
    )


    try:

        train_df = pd.read_csv(
            train_csv_path,
            encoding='utf-8'
        )


        print(
            f"Training dataset: "
            f"{len(train_df)} rows"
        )


    except Exception as e:

        print(
            f"Failed to load "
            f"training dataset: {e}"
        )


        sys.exit(1)


    try:

        val_df = pd.read_csv(
            val_csv_path,
            encoding='utf-8'
        )


        print(
            f"Validation dataset: "
            f"{len(val_df)} rows"
        )


    except Exception as e:

        print(
            f"Failed to load "
            f"validation dataset: {e}"
        )


        sys.exit(1)


    for df, name in [
        (
            train_df,
            "training dataset"
        ),
        (
            val_df,
            "validation dataset"
        )
    ]:

        if (
            smiles_column
            not in df.columns
        ):

            print(
                f"Column '{smiles_column}' "
                f"does not exist in "
                f"the {name}."
            )


            print(
                f"Available columns: "
                f"{', '.join(df.columns.tolist())}"
            )


            sys.exit(1)


        if (
            target_column
            not in df.columns
        ):

            print(
                f"Column '{target_column}' "
                f"does not exist in "
                f"the {name}."
            )


            print(
                f"Available columns: "
                f"{', '.join(df.columns.tolist())}"
            )


            sys.exit(1)


    print(
        "\n" + "=" * 70
    )


    print(
        "Loading ChemBERTa-ZINC Model"
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
        "Transformers imported successfully"
    )


    print(
        f"\nPreparing to load model: "
        f"{model_name}"
    )


    print(
        "The model will be downloaded "
        "during the first run."
    )


    try:

        print(
            "   Loading tokenizer..."
        )


        tokenizer = (
            TokenizerClass
            .from_pretrained(
                model_name,
                trust_remote_code=True
            )
        )


        print(
            "   Loading model..."
        )


        chemberta_model = (
            ModelClass
            .from_pretrained(
                model_name,
                trust_remote_code=True
            )
        )


        chemberta_model = (
            chemberta_model.to(
                device
            )
        )


        chemberta_model.eval()


        embedding_dim = (
            chemberta_model
            .config
            .hidden_size
        )


        print(
            "Model loaded successfully!"
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
            "\nPlease check your network "
            "connection or download "
            "the model manually."
        )


        sys.exit(1)


    print(
        "\n" + "=" * 70
    )


    print(
        "Extracting Embeddings"
    )


    print(
        "=" * 70
    )


    print(
        "\nProcessing training dataset..."
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
        f"   Training embeddings: "
        f"{train_embeddings.shape}"
    )


    print(
        "\nProcessing validation dataset..."
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
        f"   Validation embeddings: "
        f"{val_embeddings.shape}"
    )


    if device == "cuda":

        del chemberta_model

        torch.cuda.empty_cache()


    print(
        "\n" + "=" * 70
    )


    print(
        "Building BiLSTM Model"
    )


    print(
        "=" * 70
    )


    bilstm_model = EmbeddingBiLSTM(
        embedding_dim=embedding_dim,
        hidden_dim=256,
        lstm_hidden_1=128,
        lstm_hidden_2=256,
        lstm_hidden_3=512,
        lstm_dropout=0.3,
        fc_dropout=0.3
    ).to(device)


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
        "\nModel Parameters:"
    )


    print(
        f"   Total parameters: "
        f"{total_params:,}"
    )


    print(
        f"   Trainable parameters: "
        f"{trainable_params:,}"
    )


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


    print(
        "\n" + "=" * 70
    )


    print(
        "Final Evaluation"
    )


    print(
        "=" * 70
    )


    bilstm_model.eval()


    with torch.no_grad():

        val_X = torch.FloatTensor(
            val_embeddings
        ).to(device)


        predictions = (
            bilstm_model(
                val_X
            )
            .cpu()
            .numpy()
        )


        actual = val_labels


        final_metrics = (
            calculate_metrics(
                predictions,
                actual
            )
        )


        print(
            "Validation Set Performance:"
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
        "\nPrediction results "
        "saved to: predictions.csv"
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
        "All Tasks Completed!"
    )


    print(
        "=" * 70
    )