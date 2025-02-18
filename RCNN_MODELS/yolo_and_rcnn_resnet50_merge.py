import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import nms
import matplotlib.pyplot as plt

def show_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# Load YOLO model
yolo_model = YOLO(r"C:\Users\yadne\Desktop\24CP20\Resources\Models\YOLO_MODELS\CUSTOM\runs\detect\train3\weights\best.pt")  # Load your trained YOLO model

# Load Faster R-CNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn.to(device)
faster_rcnn.eval()

# Transform for Faster R-CNN input
transform = T.Compose([T.ToTensor()])

# Function to get proposals from YOLO
def get_yolo_proposals(image):
    results = yolo_model(image)
    proposals = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = box.conf[0].item()
            cls = int(box.cls[0].item())  # Class label
            if score > 0.5:  # Confidence threshold
                proposals.append([x1, y1, x2, y2, score, cls])
    return proposals

# Function to refine detections with Faster R-CNN
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

# Function to merge results from both models
def merge_results(yolo_boxes, frcnn_boxes, iou_threshold=0.5):
    all_boxes = torch.tensor(yolo_boxes + frcnn_boxes, dtype=torch.float32)
    if len(all_boxes) == 0:
        return []

    boxes = all_boxes[:, :4]
    scores = all_boxes[:, 4]

    keep = nms(boxes, scores, iou_threshold)
    final_boxes = all_boxes[keep].tolist()

    return final_boxes

# Start live webcam detection
cap = cv2.VideoCapture(0)  # Open webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for YOLO

    # Stage 1: YOLO detections
    yolo_proposals = get_yolo_proposals(frame_rgb)

    # Stage 2: Faster R-CNN refinement
    refined_boxes = refine_detections_with_frcnn(frame_rgb, yolo_proposals)

    # Merge results
    final_detections = merge_results(yolo_proposals, refined_boxes)

    # Draw results on the frame
    for x1, y1, x2, y2, score, label in final_detections:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Class {int(label)}: {score:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.waitKey(1)
    cv2.imshow("YOLO + Faster R-CNN Live Detection", frame)

    # Show frame
    #show_image(frame)

    #cv2.imshow("YOLO + Faster R-CNN Live Detection", frame)
    
    # Exit on pressing 'q'
    #if cv2.waitKey(1) & 0xFF == ord("q"):
        #break

cap.release()
cv2.destroyAllWindows()
