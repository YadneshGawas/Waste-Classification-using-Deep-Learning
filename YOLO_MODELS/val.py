if __name__ == "__main__":
    from ultralytics import YOLO

    # Load a model
    model = YOLO(r"C:\Users\yadne\Desktop\24CP20\Resources\Models\runs\detect\v1_15EP\weights\best.pt")

    # Validate with a custom dataset
    metrics = model.val(data=r"C:\Users\yadne\Desktop\24CP20\Resources\Models\YOLO_MODELS\dataset_v5\data.yaml")
    print(metrics.box.map)  # map50-95