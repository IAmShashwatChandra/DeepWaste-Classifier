import json
import shutil
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================= Configuration =================
TARGET_CLASSES = ["plastic", "paper_cardboard", "metal", "glass", "general_trash"]

TRASHNET_DIR = Path("raw_datasets/trashnet")
TACO_DIR = Path("raw_datasets/taco")
OUTPUT_DIR = Path("dataset_prepared")

# 70% Train, 15% Val, 15% Test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ================= Class Mappings =================
TRASHNET_MAP = {
    "plastic": "plastic",
    "paper": "paper_cardboard",
    "cardboard": "paper_cardboard",
    "metal": "metal",
    "glass": "glass",
    "trash": "general_trash",
}

TACO_NAME_MAP = {
    # Plastic
    "Clear plastic bottle": "plastic",
    "Other plastic bottle": "plastic",
    "Plastic bottle cap": "plastic",
    "Drink carton": "paper_cardboard",
    "Disposable plastic cup": "plastic",
    "Other plastic cup": "plastic",
    "Plastic lid": "plastic",
    "Bottle cap": "metal",
    "Plastic straw": "plastic",
    "Plastic bag & wrapper": "plastic",
    "Plastified paper bag": "plastic",
    "Plastic film": "plastic",
    "Six pack rings": "plastic",
    "Garbage bag": "plastic",
    "Other plastic": "plastic",
    "Polypropylene bag": "plastic",
    # Paper & Cardboard
    "Corrugated carton": "paper_cardboard",
    "Egg carton": "paper_cardboard",
    "Meal carton": "paper_cardboard",
    "Other carton": "paper_cardboard",
    "Paper cup": "paper_cardboard",
    "Paper bag": "paper_cardboard",
    "Normal paper": "paper_cardboard",
    "Magazine paper": "paper_cardboard",
    "Tissues": "paper_cardboard",
    "Wrapping paper": "paper_cardboard",
    # Metal
    "Aluminium foil": "metal",
    "Metal lid": "metal",
    "Drink can": "metal",
    "Food can": "metal",
    "Aerosol": "metal",
    "Aluminium blister pack": "metal",
    "Scrap metal": "metal",
    # Glass
    "Glass bottle": "glass",
    "Glass jar": "glass",
    "Broken glass": "glass",
    # General Trash / Unrecyclable
    "Styrofoam piece": "general_trash",
    "Foam cup": "general_trash",
    "Disposable food container": "general_trash",
    "Cigarette": "general_trash",
    "Shoe": "general_trash",
    "Battery": "general_trash",
    "Unlabeled litter": "general_trash",
}


def process_trashnet(temp_dir: Path):
    """Copies and harmonizes TrashNet images using Path methods."""
    if not TRASHNET_DIR.exists():
        print(f"Skipping TrashNet (directory not found at {TRASHNET_DIR})")
        return

    print("Processing TrashNet images...")
    img_counter = 0

    for source_class, target_class in TRASHNET_MAP.items():
        src_folder = TRASHNET_DIR / source_class
        if not src_folder.exists():
            continue

        target_folder = temp_dir / target_class
        target_folder.mkdir(parents=True, exist_ok=True)

        for img_path in src_folder.glob("*.*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                dst = target_folder / f"trashnet_{img_counter}{img_path.suffix}"
                shutil.copy2(img_path, dst)
                img_counter += 1


def locate_taco_files(base_dir: Path):
    """Resolves locations of annotations.json and base images directory."""
    candidate_annotations = [
        base_dir / "annotations.json",
        base_dir / "data" / "annotations.json",
    ]

    ann_path = None
    for cand in candidate_annotations:
        if cand.exists():
            ann_path = cand
            break

    if ann_path is None:
        return None, None

    # Base image directory is usually parent of annotations or 'data' subfolder
    img_root = ann_path.parent
    return ann_path, img_root


def process_taco(temp_dir: Path):
    """Extracts bounding box crops from TACO dataset."""
    ann_path, img_root = locate_taco_files(TACO_DIR)

    if ann_path is None or not ann_path.exists():
        print(f"Skipping TACO (annotations.json not found in {TACO_DIR} or subfolders)")
        return

    print(f"Loading annotations from: {ann_path}")
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_target = {}
    for cat in coco["categories"]:
        cat_name = cat["name"]
        target = TACO_NAME_MAP.get(cat_name)
        if target:
            cat_id_to_target[cat["id"]] = target

    img_id_to_info = {img["id"]: img for img in coco["images"]}
    crop_counter = 0

    print("Processing TACO crops...")
    for ann in tqdm(coco["annotations"]):
        cat_id = ann["category_id"]
        if cat_id not in cat_id_to_target:
            continue

        target_class = cat_id_to_target[cat_id]
        img_info = img_id_to_info.get(ann["image_id"])
        if not img_info:
            continue

        rel_file_path = img_info["file_name"]

        # Search for image under img_root, img_root/data, or TACO_DIR
        candidate_paths = [
            img_root / rel_file_path,
            img_root / "data" / rel_file_path,
            TACO_DIR / rel_file_path,
            TACO_DIR / "data" / rel_file_path,
        ]

        img_path = None
        for p in candidate_paths:
            if p.exists():
                img_path = p
                break

        if img_path is None:
            continue

        bbox = ann["bbox"]
        x, y, w, h = bbox

        # Filter out tiny or zero-size artifacts
        if w < 25 or h < 25:
            continue

        try:
            with Image.open(img_path) as im:
                im_w, im_h = im.size
                pad_w = w * 0.05
                pad_h = h * 0.05
                left = max(0, int(x - pad_w))
                upper = max(0, int(y - pad_h))
                right = min(im_w, int(x + w + pad_w))
                lower = min(im_h, int(y + h + pad_h))

                cropped = im.crop((left, upper, right, lower)).convert("RGB")
                target_folder = temp_dir / target_class
                target_folder.mkdir(parents=True, exist_ok=True)

                out_path = target_folder / f"taco_{crop_counter}.jpg"
                cropped.save(out_path, quality=95)
                crop_counter += 1
        except Exception:
            continue

    print(f"Extracted {crop_counter} object crops from TACO.")


def build_stratified_split(temp_dir: Path):
    """Splits unified images into train/val/test splits."""
    print("Building stratified splits...")
    all_images = []
    all_labels = []

    for class_name in TARGET_CLASSES:
        class_folder = temp_dir / class_name
        if not class_folder.exists():
            continue
        for img_path in class_folder.glob("*.jpg"):
            all_images.append(img_path)
            all_labels.append(class_name)

    if not all_images:
        print("No images found to split. Verify source dataset paths.")
        return

    train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
        all_images,
        all_labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=all_labels,
        random_state=42,
    )

    relative_val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
        temp_imgs,
        temp_lbls,
        test_size=(1.0 - relative_val_size),
        stratify=temp_lbls,
        random_state=42,
    )

    splits = {
        "train": (train_imgs, train_lbls),
        "val": (val_imgs, val_lbls),
        "test": (test_imgs, test_lbls),
    }

    for split_name, (imgs, lbls) in splits.items():
        for img_p, lbl in zip(imgs, lbls):
            dest = OUTPUT_DIR / split_name / lbl / img_p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_p, dest)

    shutil.rmtree(temp_dir)
    print(f"\nPreparation complete! Files structured in '{OUTPUT_DIR.resolve()}'.")
    print(
        f"Distribution -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}"
    )


if __name__ == "__main__":
    temp_pool_dir = Path("temp_pooled_dataset")
    temp_pool_dir.mkdir(parents=True, exist_ok=True)

    process_trashnet(temp_pool_dir)
    process_taco(temp_pool_dir)
    build_stratified_split(temp_pool_dir)