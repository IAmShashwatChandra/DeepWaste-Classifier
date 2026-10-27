from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ================= Configuration =================
DATA_DIR = Path("dataset_prepared")
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. Minimal Data Pipelines =================
# Minimal preprocessing without heavy augmentations (baseline setup)
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    ),
}

image_datasets = {
    split: datasets.ImageFolder(DATA_DIR / split, data_transforms[split])
    for split in ["train", "val", "test"]
}

dataloaders = {
    split: DataLoader(
        image_datasets[split],
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    for split in ["train", "val", "test"]
}

class_names = image_datasets["train"].classes
print(f"Device: {DEVICE}")
print(f"Detected Classes: {class_names}")


# ================= 2. Model Architecture =================
def build_baseline_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze feature extractor layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(in_features, num_classes)
    )
    return model


model = build_baseline_model(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)


# ================= 3. Training Loop =================
def train_model():
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(dataloaders["train"], desc="Training"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = running_loss / len(image_datasets["train"])
        epoch_train_acc = (
            running_corrects.double() / len(image_datasets["train"])
        ).item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloaders["val"], desc="Validation"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(image_datasets["val"])
        epoch_val_acc = (
            val_corrects.double() / len(image_datasets["val"])
        ).item()

        print(
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}%"
        )
        print(
            f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc * 100:.2f}%"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "resnet50_baseline_best.pth")

    print(f"\nBest Validation Accuracy: {best_val_acc * 100:.2f}%")


# ================= 4. Evaluation on Test Set =================
def evaluate_test():
    print("\nEvaluating baseline on unseen test set...")
    model.load_state_dict(
        torch.load("resnet50_baseline_best.pth", map_location=DEVICE)
    )
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders["test"], desc="Testing"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Classification Metrics
    print("\nClassification Report (Baseline):")
    print(
        classification_report(
            all_labels, all_preds, target_names=class_names, digits=4
        )
    )

    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Baseline ResNet-50 Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig("baseline_confusion_matrix.png", dpi=300)
    print("Saved confusion matrix heatmap as 'baseline_confusion_matrix.png'")


if __name__ == "__main__":
    train_model()
    evaluate_test()