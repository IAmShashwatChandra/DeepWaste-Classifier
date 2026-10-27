from ultralytics import YOLO


def main():
  model = YOLO("yolov8s.pt")
  model.train(
      data="dataset_yolo/waste_data.yaml",
      epochs=30,
      imgsz=512,  # Lowered from 640
      batch=8,  # Lowered from 16
      device=0,
      workers=0,  # Prevents Windows shared memory crashes
      patience=10,
      save=True,
      project="waste_yolo_runs",
      name="yolov8_waste_detector",
  )


if __name__ == "__main__":
  main()