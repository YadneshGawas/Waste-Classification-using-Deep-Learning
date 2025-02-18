import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision

# ✅ Load Faster R-CNN as a Region Proposal Network (RPN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn.to(device)
faster_rcnn.eval()

# ✅ Load YOLO Model for Final Object Classification
yolo_model = YOLO(r"C:\Users\yadne\Desktop\24CP20\Resources\Models\YOLO_MODELS\CUSTOM\runs\detect\train3\weights\best.pt")  # Load trained YOLO model

# 🔹 Transform for Faster R-CNN input
transform = T.Compose([T.ToTensor()])

# 🔹 Function: Get Object Proposals from Faster R-CNN
def get_frcnn_proposals(image):
    image_tensor = transform(image).to(device)
    with torch.no_grad():
        outputs = faster_rcnn([image_tensor])  # Get predictions

    proposals = []
    for i, box in enumerate(outputs[0]["boxes"]):
        score = outputs[0]["scores"][i].item()
        if score > 0.5:  # Only keep high-confidence detections
            x1, y1, x2, y2 = box.cpu().numpy()
            proposals.append([x1, y1, x2, y2])

    return proposals  # List of detected regions

# 🔹 Function: Crop Image Regions and Pass to YOLO
def classify_with_yolo(image, proposals):
    final_detections = []
    for x1, y1, x2, y2 in proposals:
        cropped_region = image[int(y1):int(y2), int(x1):int(x2)]  # Crop object region
        if cropped_region.size == 0:
            continue  # Skip if region is empty

        yolo_results = yolo_model(cropped_region)  # Pass cropped region to YOLO

        for result in yolo_results:
            for box in result.boxes:
                score = box.conf[0].item()
                cls = int(box.cls[0].item())  # Class ID from YOLO

                if score > 0.5:  # Confidence threshold for YOLO
                    final_detections.append([x1, y1, x2, y2, score, cls])

    return final_detections  # Filtered detections

# ✅ Start Webcam for Real-Time Detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # 🔹 Step 1: Get Object Proposals from Faster R-CNN
    proposals = get_frcnn_proposals(frame_rgb)

    # 🔹 Step 2: Classify & Filter with YOLO
    final_detections = classify_with_yolo(frame_rgb, proposals)

    # 🔹 Step 3: Draw Final Bounding Boxes
    for x1, y1, x2, y2, score, label in final_detections:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Class {int(label)}: {score:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show Output
    cv2.imshow("Faster R-CNN + YOLO Filtering", frame)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
