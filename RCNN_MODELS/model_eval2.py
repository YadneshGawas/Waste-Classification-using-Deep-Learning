import torch
import torchvision
import os
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.transforms import ToTensor

# ✅ Custom Dataset Class for COCO Format
class COCODataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transforms = transforms or ToTensor()

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)
        return image, img_id

    def __len__(self):
        return len(self.img_ids)

# ✅ Paths
dataset_root = r"C:\Users\yadne\Desktop\24CP20\Resources\Models\RCNN_MODELS\COCO_SET"
val_dir = os.path.join(dataset_root, "valid")
test_dir = os.path.join(dataset_root, "test")
val_json = os.path.join(val_dir, "_annotations.coco.json")
test_json = os.path.join(test_dir, "_annotations.coco.json")
model_path = r"C:\Users\yadne\Desktop\24CP20\Resources\faster_rcnn_custom.pth"

# ✅ Define DataLoader
transform = ToTensor()
val_dataset = COCODataset(val_dir, val_json, transforms=transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

test_dataset = COCODataset(test_dir, test_json, transforms=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ✅ Load Trained Faster R-CNN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = len(val_dataset.coco.cats) + 1

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ✅ Function to Evaluate Using COCO API
def evaluate_coco(model, data_loader, annotation_json, result_json):
    results = []
    model.eval()
    with torch.no_grad():
        for images, img_ids in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for img_id, output in zip(img_ids, outputs):
                boxes = output['boxes'].detach().cpu().numpy()
                scores = output['scores'].detach().cpu().numpy()
                labels = output['labels'].detach().cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    result = {
                        "image_id": int(img_id),
                        "category_id": int(label),
                        "bbox": [float(x) for x in box],
                        "score": float(score)
                    }
                    results.append(result)

    # ✅ Save predictions to JSON
    with open(result_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Saved evaluation results to {result_json}")

    # ✅ Load Ground Truth and Predictions
    coco_gt = COCO(annotation_json)
    if len(results) == 0:
        print("⚠️ No detections found! Skipping COCO evaluation.")
        return  

    coco_dt = coco_gt.loadRes(result_json)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if not hasattr(coco_eval, "stats") or len(coco_eval.stats) < 2:
        print("⚠️ Error: COCO evaluation stats not available!")
        return  

    # ✅ Extract only AP@50 and AP@50:95
    map_50_95 = coco_eval.stats[0]  # AP@[0.50:0.95]
    map_50 = coco_eval.stats[1]      # AP@0.50

    print(f"\n📊 Mean Average Precision (mAP) @50:95: {map_50_95:.4f}")
    print(f"📊 Mean Average Precision (mAP) @50: {map_50:.4f}\n")

# ✅ Run Evaluation
print("\n--- Evaluating on Validation Set ---")
val_results_json = "val_results.json"
evaluate_coco(model, val_loader, val_json, val_results_json)

print("\n--- Evaluating on Test Set ---")
test_results_json = "test_results.json"
evaluate_coco(model, test_loader, test_json, test_results_json)
