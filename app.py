from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

# ================= Configuration =================
st.set_page_config(
    page_title="Deep Waste - AI Waste Vision System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_PATH = "efficientnet_proposed.pth"
YOLO_PATH = r"runs\detect\waste_yolo_runs\yolov8_waste_detector-4\weights\best.pt"

CLASSES = ["general_trash", "glass", "metal", "paper_cardboard", "plastic"]


# ================= Resource Loaders =================
@st.cache_resource
def load_classifier():
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
  model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
  model.to(DEVICE)
  model.eval()
  return model


@st.cache_resource
def load_yolo():
  if Path(YOLO_PATH).exists():
    return YOLO(YOLO_PATH)
  return YOLO("yolov8s.pt")


# ================= Grad-CAM Engine =================
class GradCAM:

  def __init__(self, model, target_layer):
    self.model = model
    self.target_layer = target_layer
    self.gradients = None
    self.activations = None
    self._register_hooks()

  def _register_hooks(self):

    def forward_hook(module, input, output):
      self.activations = output

    def backward_hook(module, grad_in, grad_out):
      self.gradients = grad_out[0]

    self.target_layer.register_forward_hook(forward_hook)
    self.target_layer.register_full_backward_hook(backward_hook)

  def generate(self, input_tensor, target_class=None):
    self.model.zero_grad()
    output = self.model(input_tensor)

    if target_class is None:
      target_class = torch.argmax(output, dim=1).item()

    loss = output[0, target_class]
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

    probs = torch.softmax(output, dim=1)[0].cpu().detach().numpy()
    return cam, probs, target_class


classifier_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ================= Top Banner =================
def render_header():
  st.markdown(
      """
        <div style="text-align: center; padding-bottom: 15px;">
            <h1 style="font-size: 3.2rem; margin-bottom: 0px; color: #2E7D32;">♻️ Deep Waste</h1>
            <p style="font-size: 1.15rem; color: #555; margin-top: 5px;">
                Robust Computer Vision & Explainable AI for Real-World Waste Sorting
            </p>
        </div>
        <hr style="margin-top: 0px; margin-bottom: 25px;">
    """,
      unsafe_allow_html=True,
  )


# ================= Sidebar Controls =================
st.sidebar.title("🌿 Deep Waste Controls")
mode = st.sidebar.radio(
    "Select Vision Architecture:",
    [
        "EfficientNet-B2 Classifier + Grad-CAM",
        "YOLOv8 Multi-Object Detection",
        "Comprehensive Project Overview",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Runtime Specifications")
st.sidebar.info(
    f"**Inference Compute:** `{DEVICE.type.upper()}`\n\n"
    f"**Dataset Source:** TACO (Trash Annotations in Context)\n\n"
    f"**Categories (5):** Plastic, Paper/Cardboard, Metal, Glass, General Trash"
)

# ================= RENDER HEADER =================
render_header()

# ----------------- VIEW 1: Project Overview & Deep Dive -----------------
if mode == "Comprehensive Project Overview":
  col_m1, col_m2, col_m3, col_m4 = st.columns(4)
  col_m1.metric("Classifier Accuracy", "80.8%", "+9.2% over ResNet-50")
  col_m2.metric("Inference Latency", "5.7 ms", "RTX 3050 CUDA")
  col_m3.metric("Target Classes", "5", "Mapped from 60 TACO categories")
  col_m4.metric("Detection Backbone", "YOLOv8s", "Anchor-Free Multi-Scale")

  st.markdown("---")

  st.header("1. Problem Statement & Motivation")
  st.write(
      "Municipal recycling facilities face acute operational bottlenecks due to"
      " manual sorting errors, worker health hazards, and cross-contamination"
      " in recyclables. Standard computer vision models encounter distinct"
      " failure modes when deployed in uncontrolled municipal environments:"
  )
  st.markdown("""
    - **Background Correlation Bias:** Traditional CNNs frequently latch onto background textures (grass, pavement, sand, floor tiles) rather than the waste artifact itself.
    - **Extreme Class Imbalance:** Datasets are heavily dominated by single-use plastics and general litter, leaving metals and glassware with low recall.
    - **Single-Label Bottlenecks:** Real-world dump sites do not feature isolated items; they contain dense, overlapping piles with multiple recyclable materials mixed together.
    """)

  st.header("2. Comparative Architectural Benchmarks")
  st.write(
      "To resolve these challenges, **Deep Waste** incorporates a dual-stage"
      " pipeline moving from robust classification to multi-object spatial"
      " localization:"
  )

  st.table({
      "Architecture": [
          "ResNet-50 (Baseline)",
          "EfficientNet-B2 (Proposed Classifier)",
          "YOLOv8s (Multi-Object Detector)",
      ],
      "Loss Function / Objective": [
          "Cross-Entropy Loss",
          "Focal Loss (gamma=2.0, alpha-weighted)",
          "Task-Aligned Loss (CIoU + DFL + BCE)",
      ],
      "Augmentations": [
          "Standard Resizing / Horizontal Flip",
          "Albumentations (ColorJitter, GaussNoise, ShiftScaleRotate)",
          "Mosaic, MixUp, HSV jitter",
      ],
      "Validation Metric": [
          "71.6% Accuracy",
          "80.8% Accuracy",
          "25.8% mAP50 (Dense Uncontrolled Piles)",
      ],
      "Primary Deliverable": [
          "Baseline benchmark",
          "Material classification & visual saliency",
          "Simultaneous multi-item localization & counting",
      ],
  })

  st.header("3. Mathematical Formulations")

  st.subheader("Focal Loss Optimization")
  st.write(
      "Standard cross-entropy loss provides uniform gradients regardless of"
      " sample difficulty. To counter class imbalance and focus the network on"
      " hard metal and glass contours, Focal Loss is applied:"
  )
  st.latex(r"\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)")
  st.caption(
      "Where $p_t$ represents the model's estimated probability for the"
      " ground-truth class, $\\alpha_t$ balances underrepresented classes, and"
      " $\\gamma = 2.0$ downweights easily classified background examples."
  )

  st.subheader("Gradient-Weighted Class Activation Mapping (Grad-CAM)")
  st.write(
      "To verify whether the classifier focuses on genuine material features"
      " rather than background artifacts, neuron importance weights"
      " $\\alpha_k^c$ are computed via global average pooling of gradients"
      " flowing into final convolutional layer $A^k$:"
  )
  st.latex(
      r"\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial"
      r" A_{i,j}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k"
      r" \alpha_k^c A^k\right)"
  )

  st.stop()

# ----------------- INFERENCE WORKFLOW (CLASSIFICATION & DETECTION) -----------------
uploaded_file = st.sidebar.file_uploader(
    "Upload Waste Image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
  st.info("👈 Upload an image from the sidebar to inspect predictions.")

  with st.expander("ℹ️ How to Use Deep Waste", expanded=True):
    st.markdown("""
        1. **Select Architecture Mode** from the sidebar:
           - **EfficientNet-B2 Classifier + Grad-CAM:** Ideal for single items, material classification, and auditing model attention hotspots.
           - **YOLOv8 Multi-Object Detection:** Ideal for cluttered piles containing multiple waste items that need to be counted and bounded.
        2. **Upload a clear photo** (supports mobile captures, JPG, PNG).
        3. **Inspect Output:** Review class distributions, Grad-CAM heatmaps, or localized bounding box metrics.
        """)
  st.stop()

# Load and downscale safely to avoid OpenCV allocation crashes
pil_image = Image.open(uploaded_file).convert("RGB")
pil_image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
img_np = np.array(pil_image)

col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
  st.subheader("Input Image Stream")
  st.image(
      pil_image,
      caption=f"Dimensions: {img_np.shape[1]}x{img_np.shape[0]} px",
      use_container_width=True,
  )

  with st.expander("🖼️ Input Metadata"):
    st.write(f"- **Aspect Ratio:** {img_np.shape[1] / img_np.shape[0]:.2f}")
    st.write(f"- **Color Channels:** RGB ({img_np.shape[2]})")
    st.write(f"- **Host Device:** {DEVICE.type.upper()}")

with col_right:
  # ----------------- MODE 2: YOLOv8 Object Detection -----------------
  if mode == "YOLOv8 Multi-Object Detection":
    st.subheader("Multi-Object Detection Results")
    st.caption(
        "Detects and localizes multiple simultaneous recyclable entities within"
        " cluttered piles."
    )

    conf_slider = st.sidebar.slider(
        "Confidence Threshold", 0.05, 0.90, 0.20, 0.05
    )
    iou_slider = st.sidebar.slider("IoU NMS Threshold", 0.10, 0.80, 0.45, 0.05)

    yolo = load_yolo()
    results = yolo.predict(
        source=img_np,
        conf=conf_slider,
        iou=iou_slider,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )

    res_plot = results[0].plot()
    res_plot_rgb = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)
    st.image(
        res_plot_rgb,
        caption="YOLOv8 Detection Output",
        use_container_width=True,
    )

    boxes = results[0].boxes
    if len(boxes) > 0:
      st.markdown(f"### Detected Entities: `{len(boxes)}`")

      counts = {}
      for box in boxes:
        cid = int(box.cls[0].item())
        cname = yolo.names[cid].replace("_", " ").title()
        counts[cname] = counts.get(cname, 0) + 1

      cnt_cols = st.columns(len(counts))
      for idx, (k, v) in enumerate(counts.items()):
        cnt_cols[idx].metric(k, f"{v} items")

      st.markdown("#### Bounding Box Inventory")
      for idx, box in enumerate(boxes):
        cid = int(box.cls[0].item())
        label = yolo.names[cid].replace("_", " ").title()
        score = float(box.conf[0].item())
        coords = [round(c, 1) for c in box.xyxy[0].tolist()]
        st.write(
            f"- **Item {idx + 1}:** `{label}` — **{score * 100:.1f}%**"
            f" confidence | BBox: `{coords}`"
        )
    else:
      st.warning(
          "No objects detected above threshold. Lower the threshold slider in"
          " the sidebar."
      )

    with st.expander("📖 Technical Notes on YOLOv8 Detection"):
      st.markdown("""
            - **Spatial Decoupling:** YOLOv8 predicts bounding boxes independently per cell, bypassing the single-label classification restriction.
            - **NMS (Non-Maximum Suppression):** The `IoU NMS` slider controls how overlapping bounding boxes targeting the same physical object are eliminated.
            - **Inference Speed:** Runs at ~5.7ms per frame on local GPU hardware, enabling real-time conveyor belt sorting automation.
            """)

  # ----------------- MODE 3: Classifier + Grad-CAM -----------------
  else:
    st.subheader("Classification & Visual Saliency")
    st.caption(
        "EfficientNet-B2 material identification coupled with Grad-CAM"
        " explainability."
    )

    if not Path(CLASSIFIER_PATH).exists():
      st.error(
          f"Checkpoint `{CLASSIFIER_PATH}` not found in root. Train the"
          " classifier first."
      )
      st.stop()

    clf = load_classifier()
    gradcam = GradCAM(clf, clf.features[-1])

    processed = (
        classifier_transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)
    )

    with torch.no_grad():
      initial_logits = clf(processed)
      initial_probs = torch.softmax(initial_logits, dim=1)[0].cpu().numpy()
      top_class_idx = int(np.argmax(initial_probs))

    top_class_name = CLASSES[top_class_idx].replace("_", " ").title()
    top_confidence = initial_probs[top_class_idx]

    st.metric(
        label="Primary Material Prediction",
        value=top_class_name,
        delta=f"{top_confidence * 100:.2f}% Confidence",
    )

    st.markdown("#### Visual Saliency Class Selector")
    selected_explain_class = st.selectbox(
        "Select target class to inspect via Grad-CAM:",
        options=CLASSES,
        index=top_class_idx,
        format_func=lambda x: (
            f"{x.replace('_', ' ').title()} (Confidence:"
            f" {initial_probs[CLASSES.index(x)]*100:.1f}%)"
        ),
    )
    target_idx = CLASSES.index(selected_explain_class)

    cam, probs, _ = gradcam.generate(processed, target_class=target_idx)

    st.markdown("#### Softmax Confidence Breakdown")
    for cname, pval in zip(CLASSES, probs):
      disp = cname.replace("_", " ").title()
      st.write(f"**{disp}**: `{pval * 100:.1f}%`")
      st.progress(float(pval))

    st.markdown(
        f"#### Attention Heatmap: **{selected_explain_class.replace('_', ' ').title()}**"
    )
    resized_orig = cv2.resize(img_np, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(resized_orig, 0.6, heatmap, 0.4, 0)

    cam_c1, cam_c2 = st.columns(2)
    with cam_c1:
      st.image(cam, caption="Raw Feature Activation", use_container_width=True)
    with cam_c2:
      st.image(
          overlay, caption="Superimposed Grad-CAM", use_container_width=True
      )

    with st.expander("📖 Interpreting the Grad-CAM Heatmap"):
      st.markdown("""
            - **🔴 Red / Orange:** Peak activation zones. The features here (reflective rim, crinkles, transparency) provided the strongest evidence for the selected class.
            - **🟢 Green / Yellow:** Moderate supporting context.
            - **🔵 Deep Blue:** Neutral zones completely ignored by the model (e.g. background grass, ground, shadows).
            - **Multi-Object Diagnostic:** When you switch the dropdown from `Plastic` to `Paper Cardboard`, observe how the focal spot shifts from bags to cartons.
            """)