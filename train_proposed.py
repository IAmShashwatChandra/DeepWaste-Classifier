from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

# ================= Configuration =================
DATA_DIR = Path("dataset_prepared")
BATCH_SIZE = 32
HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 20
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. Albumentations Pipelines =================
train_transforms = A.Compose(
    [
        A.Resize(260, 260),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5
        ),
        A.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5
        ),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ],
            p=0.3,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(8, 24),
            hole_width_range=(8, 24),
            p=0.3,
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

eval_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


class WasteDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.classes = sorted(
            [d.name for d in root_dir.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            for img_p in (root_dir / cls).glob("*.jpg"):
                self.image_paths.append(img_p)
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_p = self.image_paths[idx]
        image = np.array(Image.open(img_p).convert("RGB"))
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


train_dataset = WasteDataset(DATA_DIR / "train", transform=train_transforms)
val_dataset = WasteDataset(DATA_DIR / "val", transform=eval_transforms)
test_dataset = WasteDataset(DATA_DIR / "test", transform=eval_transforms)

class_names = train_dataset.classes

dataloaders = {
    "train": DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    ),
    "val": DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    ),
    "test": DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    ),
}


# ================= 2. Focal Loss Implementation =================
class MulticlassFocalLoss(nn.Module):

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", weight=self.weight
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


criterion = MulticlassFocalLoss(gamma=2.0)


# ================= 3. Model Architecture =================
def build_proposed_model(num_classes):
    model = models.efficientnet_b2(
        weights=models.EfficientNet_B2_Weights.DEFAULT
    )
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model


model = build_proposed_model(NUM_CLASSES).to(DEVICE)


# ================= 4. Two-Phase Training =================
def train_loop(
    epochs,
    optimizer,
    scheduler=None,
    best_acc=0.0,
    save_name="efficientnet_proposed.pth",
):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss, train_correct = 0.0, 0

        for inputs, labels in tqdm(dataloaders["train"], desc="Train"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_correct += torch.sum(preds == labels.data)

        if scheduler:
            scheduler.step()

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(dataloaders["val"], desc="Val"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_correct += torch.sum(preds == labels.data)

        tr_acc = (train_correct.double() / len(train_dataset)).item() * 100
        vl_acc = (val_correct.double() / len(val_dataset)).item() * 100
        print(
            f"Train Acc: {tr_acc:.2f}% | Val Acc: {vl_acc:.2f}% (Loss: {val_loss / len(val_dataset):.4f})"
        )

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), save_name)
    return best_acc


def run_training():
    # Phase 1: Train Head Only
    print("\n--- Phase 1: Training Classification Head ---")
    for param in model.features.parameters():
        param.requires_grad = False

    opt_head = torch.optim.AdamW(
        model.classifier.parameters(), lr=1e-3, weight_decay=1e-2
    )
    best_acc = train_loop(HEAD_EPOCHS, opt_head)

    # Phase 2: Unfreeze Upper Blocks & Fine-tune
    print("\n--- Phase 2: Fine-Tuning Top Backbone Blocks ---")
    for param in model.features[5:].parameters():
        param.requires_grad = True

    opt_tune = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-3,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_tune, T_max=FINE_TUNE_EPOCHS, eta_min=1e-6
    )
    train_loop(
        FINE_TUNE_EPOCHS, opt_tune, scheduler=sched, best_acc=best_acc
    )


# ================= 5. Final Evaluation =================
def evaluate_test():
    print("\nEvaluating Proposed Model on Test Set...")
    model.load_state_dict(
        torch.load("efficientnet_proposed.pth", map_location=DEVICE)
    )
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders["test"], desc="Test"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report (Proposed Model):")
    print(
        classification_report(
            all_labels, all_preds, target_names=class_names, digits=4
        )
    )

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Proposed EfficientNet-B2 Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig("proposed_confusion_matrix.png", dpi=300)
    print("Saved proposed confusion matrix as 'proposed_confusion_matrix.png'")


if __name__ == "__main__":
    run_training()
    evaluate_test()