import json
from pathlib import Path
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

TACO_DIR = Path("raw_datasets/taco")
ANN_FILE = TACO_DIR / "data" / "annotations.json"
YOLO_DIR = Path("dataset_yolo")

# Map TACO categories to your 5 target classes
CATEGORY_MAP = {
    # Plastic
    "Bottle": 0,
    "Plastic bottle cap": 0,
    "Plastic bag & wrapper": 0,
    "Plastic container": 0,
    "Disposable plastic cup": 0,
    "Clear plastic bottle": 0,
    "Other plastic": 0,
    # Paper & Cardboard
    "Paper": 1,
    "Cardboard": 1,
    "Corrugated carton": 1,
    "Egg carton": 1,
    "Paper cup": 1,
    "Magazine paper": 1,
    # Metal
    "Drink can": 2,
    "Food can": 2,
    "Aerosol": 2,
    "Aluminium foil": 2,
    "Metal bottle cap": 2,
    "Other metal": 2,
    # Glass
    "Glass bottle": 3,
    "Broken glass": 3,
    "Glass jar": 3,
    # General Trash
    "Cigarette": 4,
    "Styrofoam piece": 4,
    "Unlabeled litter": 4,
}

CLASS_NAMES = [
    "plastic",
    "paper_cardboard",
    "metal",
    "glass",
    "general_trash",
]

with open(ANN_FILE, "r") as f:
    coco_data = json.load(f)

cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

# Group bounding box annotations by image ID
img_annotations = {}
for ann in coco_data["annotations"]:
    img_id = ann["image_id"]
    cat_name = cat_id_to_name.get(ann["category_id"])

    # Match class
    target_class_id = None
    for key, cid in CATEGORY_MAP.items():
        if cat_name and key.lower() in cat_name.lower():
            target_class_id = cid
            break

    if target_class_id is None:
        target_class_id = 4  # default to general_trash

    img_annotations.setdefault(img_id, []).append(
        (target_class_id, ann["bbox"])
    )

images_info = [
    img for img in coco_data["images"] if img["id"] in img_annotations
]
train_imgs, val_imgs = train_test_split(
    images_info, test_size=0.2, random_state=42
)

# Prepare folder structure
for split in ["train", "val"]:
    (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def process_split(img_list, split_name):
    for img_dict in tqdm(img_list, desc=f"Converting {split_name}"):
        src_path = TACO_DIR / "data" / img_dict["file_name"]
        if not src_path.exists():
            continue

        w_img, h_img = img_dict["width"], img_dict["height"]
        dst_img_name = f"{src_path.stem}_{img_dict['id']}.jpg"
        dst_img_path = YOLO_DIR / "images" / split_name / dst_img_name
        dst_lbl_path = (
            YOLO_DIR
            / "labels"
            / split_name
            / f"{src_path.stem}_{img_dict['id']}.txt"
        )

        shutil.copy(src_path, dst_img_path)

        # Convert COCO bbox [x_min, y_min, width, height] to YOLO [x_center, y_center, w, h] (normalized)
        lines = []
        for class_id, (bx, by, bw, bh) in img_annotations[img_dict["id"]]:
            x_center = (bx + bw / 2.0) / w_img
            y_center = (by + bh / 2.0) / h_img
            norm_w = bw / w_img
            norm_h = bh / h_img

            # Clamp boundaries
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )

        with open(dst_lbl_path, "w") as lf:
            lf.write("\n".join(lines))


process_split(train_imgs, "train")
process_split(val_imgs, "val")

# Generate the YOLO dataset configuration file
yaml_content = f"""path: {YOLO_DIR.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: plastic
  1: paper_cardboard
  2: metal
  3: glass
  4: general_trash
"""

with open(YOLO_DIR / "waste_data.yaml", "w") as f:
    f.write(yaml_content)

print(
    f"YOLO dataset setup complete! Configuration saved to {YOLO_DIR / 'waste_data.yaml'}"
)