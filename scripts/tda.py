from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import TensorDataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = DATA_DIR / "speech_features_normalized.csv"

OUTPUT_FILE = DATA_DIR / "speech_features_domain_adapted.csv"
MODEL_FILE = DATA_DIR / "domain_adversarial_model.pt"


EMOTIONS = [
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad"
]


BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
LAMBDA_DOMAIN = 0.5
RANDOM_STATE = 42


class GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambda_value):
        ctx.lambda_value = lambda_value
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_value * grad_output, None


class FeatureExtractor(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


class EmotionClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        return self.network(x)


class DomainClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


class DomainAdversarialNetwork(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.feature_extractor = FeatureExtractor(input_dim)
        self.emotion_classifier = EmotionClassifier()
        self.domain_classifier = DomainClassifier()

    def forward(self, x, lambda_domain=0.5):

        features = self.feature_extractor(x)

        emotion_output = self.emotion_classifier(features)

        reversed_features = GradientReversal.apply(
            features,
            lambda_domain
        )

        domain_output = self.domain_classifier(
            reversed_features
        )

        return emotion_output, domain_output, features


def load_data():

    df = pd.read_csv(INPUT_FILE)

    df = df[
        df["emotion"].isin(EMOTIONS)
    ].copy()

    df = df.dropna()

    metadata_columns = [
        "file_id",
        "dataset",
        "file",
        "speaker",
        "emotion",
        "emotion_id"
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    X = df[feature_columns].values.astype(np.float32)

    emotion_encoder = LabelEncoder()

    y_emotion = emotion_encoder.fit_transform(
        df["emotion"]
    )

    domain_encoder = LabelEncoder()

    y_domain = domain_encoder.fit_transform(
        df["dataset"]
    )

    return (
        df,
        X,
        y_emotion,
        y_domain,
        feature_columns,
        emotion_encoder,
        domain_encoder
    )


def create_loaders(X, y_emotion, y_domain):

    indices = np.arange(len(X))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_emotion
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    emotion_train = y_emotion[train_idx]
    emotion_test = y_emotion[test_idx]

    domain_train = y_domain[train_idx]
    domain_test = y_domain[test_idx]

    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(emotion_train),
        torch.tensor(domain_train)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(emotion_test),
        torch.tensor(domain_test)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return (
        train_loader,
        test_loader,
        train_idx,
        test_idx
    )


def evaluate(
    model,
    loader,
    device,
    emotion_encoder
):

    model.eval()

    emotion_true = []
    emotion_pred = []

    domain_true = []
    domain_pred = []

    with torch.no_grad():

        for X, y_emotion, y_domain in loader:

            X = X.to(device)

            emotion_output, domain_output, _ = model(
                X,
                lambda_domain=0.0
            )

            emotion_predictions = (
                torch.argmax(
                    emotion_output,
                    dim=1
                )
                .cpu()
                .numpy()
            )

            domain_predictions = (
                torch.argmax(
                    domain_output,
                    dim=1
                )
                .cpu()
                .numpy()
            )

            emotion_true.extend(
                y_emotion.numpy()
            )

            emotion_pred.extend(
                emotion_predictions
            )

            domain_true.extend(
                y_domain.numpy()
            )

            domain_pred.extend(
                domain_predictions
            )

    emotion_accuracy = accuracy_score(
        emotion_true,
        emotion_pred
    )

    emotion_f1 = f1_score(
        emotion_true,
        emotion_pred,
        average="macro"
    )

    domain_accuracy = accuracy_score(
        domain_true,
        domain_pred
    )

    print("\nEmotion accuracy:", emotion_accuracy)
    print("Emotion macro F1:", emotion_f1)

    print("\nEmotion classification report:")
    print(
        classification_report(
            emotion_true,
            emotion_pred,
            target_names=emotion_encoder.classes_
        )
    )

    print(
        "Domain classification accuracy:",
        domain_accuracy
    )

    return (
        emotion_accuracy,
        emotion_f1,
        domain_accuracy
    )


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    emotion_encoder
):

    emotion_loss_function = nn.CrossEntropyLoss()
    domain_loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0
        total_emotion_loss = 0
        total_domain_loss = 0

        for X, y_emotion, y_domain in train_loader:

            X = X.to(device)
            y_emotion = y_emotion.to(device)
            y_domain = y_domain.to(device)

            optimizer.zero_grad()

            emotion_output, domain_output, _ = model(
                X,
                lambda_domain=LAMBDA_DOMAIN
            )

            emotion_loss = emotion_loss_function(
                emotion_output,
                y_emotion
            )

            domain_loss = domain_loss_function(
                domain_output,
                y_domain
            )

            loss = (
                emotion_loss
                + LAMBDA_DOMAIN * domain_loss
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_emotion_loss += emotion_loss.item()
            total_domain_loss += domain_loss.item()

        if (
            epoch == 0
            or (epoch + 1) % 5 == 0
            or epoch == EPOCHS - 1
        ):

            print(
                f"\nEpoch {epoch + 1}/{EPOCHS}"
            )

            print(
                "Loss:",
                total_loss / len(train_loader)
            )

            print(
                "Emotion loss:",
                total_emotion_loss / len(train_loader)
            )

            print(
                "Domain loss:",
                total_domain_loss / len(train_loader)
            )

            evaluate(
                model,
                test_loader,
                device,
                emotion_encoder
            )


def extract_adapted_features(
    model,
    X,
    device
):

    model.eval()

    dataset = TensorDataset(
        torch.tensor(X)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    features = []

    with torch.no_grad():

        for (batch,) in loader:

            batch = batch.to(device)

            encoded = model.feature_extractor(
                batch
            )

            features.append(
                encoded.cpu().numpy()
            )

    return np.vstack(features)


def save_features(
    df,
    adapted_features,
    test_indices,
    feature_columns
):

    output = df.copy()

    for i in range(adapted_features.shape[1]):

        output[
            f"adapted_feature_{i + 1}"
        ] = adapted_features[:, i]

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved adapted features:")
    print(OUTPUT_FILE)

    print(
        "\nAdapted feature shape:",
        adapted_features.shape
    )


def run():

    print("=" * 60)
    print("DOMAIN ADVERSARIAL TRAINING")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    (
        df,
        X,
        y_emotion,
        y_domain,
        feature_columns,
        emotion_encoder,
        domain_encoder
    ) = load_data()

    print(
        "\nFiltered dataset shape:",
        df.shape
    )

    print(
        "\nEmotion distribution:"
    )

    print(
        df["emotion"].value_counts()
    )

    print(
        "\nDomain distribution:"
    )

    print(
        df["dataset"].value_counts()
    )

    print(
        "\nInput features:",
        len(feature_columns)
    )

    (
        train_loader,
        test_loader,
        train_idx,
        test_idx
    ) = create_loaders(
        X,
        y_emotion,
        y_domain
    )

    model = DomainAdversarialNetwork(
        input_dim=X.shape[1]
    ).to(device)

    print(
        "\nModel parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    print("\nStarting training...")

    train_model(
        model,
        train_loader,
        test_loader,
        device,
        emotion_encoder
    )

    print("\nExtracting adapted features...")

    adapted_features = extract_adapted_features(
        model,
        X,
        device
    )

    save_features(
        df,
        adapted_features,
        test_idx,
        feature_columns
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "emotion_classes": list(
                emotion_encoder.classes_
            ),
            "domain_classes": list(
                domain_encoder.classes_
            )
        },
        MODEL_FILE
    )

    print(
        "\nSaved model:"
    )

    print(
        MODEL_FILE
    )

    print("\n" + "=" * 60)
    print("DOMAIN ADVERSARIAL TRAINING COMPLETE")
    print("=" * 60)

    return model, df, adapted_features


if __name__ == "__main__":
    run()