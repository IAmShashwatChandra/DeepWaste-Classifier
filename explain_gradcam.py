from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["general_trash", "glass", "metal", "paper_cardboard", "plastic"]


# Load the trained model
def load_model(weights_path="efficientnet_proposed.pth"):
    model = models.efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p=0.3),
        nn.Linear(256, len(CLASSES)),
    )
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# Hook-based Grad-CAM implementation
class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
        return cam, class_idx, torch.softmax(output, dim=1)[0, class_idx].item()


def visualize(image_path, model, cam_generator, save_name="gradcam_result.png"):
    raw_img = Image.open(image_path).convert("RGB")
    transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]
    )

    img_np = np.array(raw_img)
    processed = transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)

    cam, pred_idx, conf = cam_generator.generate(processed)

    resized_orig = cv2.resize(img_np, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.6 * resized_orig + 0.4 * heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(resized_orig)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM Activation")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Pred: {CLASSES[pred_idx]} ({conf * 100:.1f}%)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"Saved {save_name} | Predicted: {CLASSES[pred_idx]} ({conf:.2f})")


if __name__ == "__main__":
    model = load_model()
    # Target the last convolutional block in EfficientNet-B2
    target_layer = model.features[-1]
    cam_gen = GradCAM(model, target_layer)

    # Test sample from test dataset
    test_samples = list(Path("dataset_prepared/test").glob("*/*.jpg"))[:5]
    for idx, sample_path in enumerate(test_samples):
        visualize(
            sample_path,
            model,
            cam_gen,
            save_name=f"gradcam_sample_{idx + 1}.png",
        )