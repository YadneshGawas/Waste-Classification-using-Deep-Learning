import torch
import cv2
import torchvision.transforms as T
from ultralytics import YOLO
import torchvision

# ✅ Load Trained YOLO Model
yolo_model = YOLO(r"C:\Users\yadne\Desktop\24CP20\Resources\Models\YOLO_MODELS\CUSTOM\runs\detect\train3\weights\best.pt")  # Replace with your YOLO model path

# ✅ Load Pretrained Faster R-CNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn.eval().to(device)

# 🔹 Function: Get YOLO Detections
def get_yolo_proposals(frame):
    results = yolo_model(frame)
    proposals = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int
            score = float(box.conf[0])  # Confidence score
            proposals.append([x1, y1, x2, y2, score])

    return proposals  # List of bounding boxes

# 🔹 Function: Refine YOLO Detections with Faster R-CNN
def refine_detections_with_frcnn(frame, yolo_proposals):
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = faster_rcnn(img_tensor)

    refined_boxes = []
    for i, box in enumerate(predictions[0]["boxes"].cpu().numpy()):
        score = predictions[0]["scores"][i].item()
        refined_boxes.append([box[0], box[1], box[2], box[3], score])

    return refined_boxes  # List of refined bounding boxes

# 🔹 Function: Merge YOLO and Faster R-CNN Results
def merge_results(yolo_proposals, refined_boxes):
    final_detections = []

    for yolo_box in yolo_proposals:
        x1, y1, x2, y2, score = yolo_box
        for refined_box in refined_boxes:
            rx1, ry1, rx2, ry2, r_score = refined_box

            # If Faster R-CNN has high confidence, use its result
            if r_score > 0.8:
                final_detections.append(refined_box)
            else:
                final_detections.append(yolo_box)

    return final_detections

# ✅ Open Webcam for Real-Time Detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔹 Step 1: Get YOLO Detections
    yolo_proposals = get_yolo_proposals(frame)

    # 🔹 Step 2: Refine Detections with Faster R-CNN
    refined_boxes = refine_detections_with_frcnn(frame, yolo_proposals)

    # 🔹 Step 3: Merge Results
    final_detections = merge_results(yolo_proposals, refined_boxes)

    # 🔹 Step 4: Draw Final Bounding Boxes
    for x1, y1, x2, y2, score in final_detections:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {score:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show Output
    cv2.imshow("YOLO + Faster R-CNN Live Detection", frame)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
