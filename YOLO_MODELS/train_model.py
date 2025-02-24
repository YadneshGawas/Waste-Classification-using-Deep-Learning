if __name__ == "__main__":
    from ultralytics import YOLO
    import torch

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = 0  # Use GPU (device index 0)
        print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"  # Use CPU
        print("CUDA not available. Using CPU.")

    # Load the model (replace 'yolo11n.pt' with your model)
    model = YOLO(r"C:\Users\yadne\Desktop\24CP20\Resources\Models\YOLO_MODELS\yolo11n.pt")

    # Train the model
    results = model.train(
    data=r"C:\Users\yadne\Desktop\24CP20\Resources\Models\YOLO_MODELS\dataset_v1\data.yaml", epochs=20, imgsz=640, batch=16, device=0, workers=8, 
    pretrained=True, optimizer="auto", verbose=True, seed=0, deterministic=True, 
    single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, 
    fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, 
    mask_ratio=4, dropout=0.0, val=True, split="val", save_json=False, save_hybrid=False, 
    conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, vid_stride=1, 
    stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, 
    retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, 
    save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, 
    line_width=None, format="torchscript", keras=False, optimize=False, int8=False, 
    dynamic=False, simplify=True, opset=None, workspace=None, nms=False, 
    lr0=0.00021837943437872777, lrf=0.01, momentum=0.7555433165480429, 
    weight_decay=0.0024367818584150134, warmup_epochs=3.0, warmup_momentum=0.8, 
    warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, 
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, 
    shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, 
    copy_paste=0.0, copy_paste_mode="flip", auto_augment="randaugment", erasing=0.4, 
    crop_fraction=1.0, tracker="botsort.yaml"
)
