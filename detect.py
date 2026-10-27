from pathlib import Path
from ultralytics import YOLO

# 1. Direct path to your newly trained weights
WEIGHTS_PATH = r"runs\detect\waste_yolo_runs\yolov8_waste_detector-4\weights\best.pt"

print(f"Loading trained weights from: {WEIGHTS_PATH}")
model = YOLO(WEIGHTS_PATH)

# 2. Target image (make sure this image exists in your waste_project root)
test_image = r"dataset_yolo\images\val"

if not Path(test_image).exists():
  print(f"\n[!] Missing image: Please place '{test_image}' in your project root.")
else:
  print(f"\nRunning detection on {test_image}...")
  results = model.predict(
      source=test_image,
      conf=0.20,  # Confidence threshold
      save=True,  # Saves bounding box visualization
  )

  print("\n--- Detections Found ---")
  for r in results:
    for box in r.boxes:
      cls_id = int(box.cls[0].item())
      conf = float(box.conf[0].item())
      cls_name = model.names[cls_id]
      print(f"- {cls_name}: {conf * 100:.1f}% confidence")

  print(
      f"\nInference complete! Visual result saved to: runs\\detect\\predict\\"
  )