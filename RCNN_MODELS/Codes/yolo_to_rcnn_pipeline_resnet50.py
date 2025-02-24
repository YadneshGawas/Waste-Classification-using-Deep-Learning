import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import nms

# ✅ Load YOLO model
yolo_model = YOLO(r"C:\Users\yadne\Desktop\24CP20\Resources\Models\YOLO_MODELS\CUSTOM\runs\detect\train3\weights\best.pt")

# ✅ Load Faster R-CNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()

# ✅ Transform for Faster R-CNN input
transform = T.Compose([T.ToTensor()])

# 🔹 Function: Get YOLO Detections
def get_yolo_proposals(image):
    results = yolo_model(image)
    proposals = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = box.conf[0].item()
            cls = int(box.cls[0].item())  # Class label
            if score > 0.8:  # Confidence threshold
                proposals.append([x1, y1, x2, y2, score, cls])
    return proposals

# 🔹 Function: Refine YOLO Detections with Faster R-CNN
def refine_detections_with_frcnn(image, proposals):
    image_tensor = transform(image).to(device)
    with torch.no_grad():
        outputs = faster_rcnn([image_tensor])  # Get Faster R-CNN predictions

    refined_boxes = []
    for i, box in enumerate(outputs[0]["boxes"]):
        score = outputs[0]["scores"][i].item()
        label = outputs[0]["labels"][i].item()
        if score > 0.5:
            x1, y1, x2, y2 = box.cpu().numpy()
            refined_boxes.append([x1, y1, x2, y2, score, label])

    return refined_boxes

# 🔹 Function: Compute IoU (Intersection over Union)
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area else 0

# 🔹 Function: Merge Results with IoU Filtering
def merge_results(yolo_boxes, frcnn_boxes, iou_threshold=0.6):
    final_detections = []
    for yolo_box in yolo_boxes:
        x1, y1, x2, y2, score, cls = yolo_box
        matched = False

        for refined_box in frcnn_boxes:
            rx1, ry1, rx2, ry2, r_score, r_cls = refined_box
            if iou((x1, y1, x2, y2), (rx1, ry1, rx2, ry2)) > iou_threshold and r_score > 0.6: #Set the RSCORE and IOU
                final_detections.append(refined_box)
                matched = True
                break  # Only take the best match

        if not matched:
            final_detections.append(yolo_box)  # Keep YOLO detection if not refined

    return final_detections

# ✅ Open Webcam for Real-Time Detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert for YOLO

    # 🔹 Step 1: Get YOLO Detections
    yolo_proposals = get_yolo_proposals(frame_rgb)

    # 🔹 Step 2: Refine Detections with Faster R-CNN
    refined_boxes = refine_detections_with_frcnn(frame_rgb, yolo_proposals)

    # 🔹 Step 3: Merge Results
    final_detections = merge_results(yolo_proposals, refined_boxes)

    # 🔹 Step 4: Draw Final Bounding Boxes
    for x1, y1, x2, y2, score, label in final_detections:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Class {int(label)}: {score:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO + Faster R-CNN Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit on pressing 'q'
        break

cap.release()
cv2.destroyAllWindows()
