import json
import os
import cv2
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_classes = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", 
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def yolo2coco_gt(image_storage_path, yolo_storage_path, coco_file_path, classes):
    coco_data = {"images": [], "annotations": [], "categories": []}
    
    for index, coco_clas in enumerate(coco_classes):
        coco_data["categories"].append({"id": index, "name": coco_clas})
    
    id = 0
    for index, image_file_name in enumerate(sorted(os.listdir(image_storage_path))):
        image_file_path = Path(os.path.join(image_storage_path, image_file_name))
        yolo_file_path = os.path.join(yolo_storage_path, image_file_path.stem + ".txt")
        
        height, width, _ = cv2.imread(image_file_path).shape
        
        coco_data["images"].append({
            "id": index, 
            "file_name": image_file_name, 
            "width": width, 
            "height": height
        })

        if not os.path.exists(yolo_file_path): continue
        
        with open(yolo_file_path, "r") as file:
            for line in file:
                clas, cx, cy, w, h = map(float, line.strip().split())
                cx *= width; cy *= height
                w *= width; h *= height

                coco_data["annotations"].append({
                    "id": id, 
                    "image_id": index, 
                    "category_id": int(clas), 
                    "bbox": [cx-w/2, cy-h/2, w, h], 
                    "area": width * height, 
                    "iscrowd": 0
                })
                id += 1
    
    with open(coco_file_path, "w") as file:
        json.dump(coco_data, file, indent=4) 

def yolo2coco_dt(image_storage_path, yolo_storage_path, coco_file_path, classes):
    coco_data = list()
    
    for index, image_file_name in enumerate(sorted(os.listdir(image_storage_path))):
        image_file_path = Path(os.path.join(image_storage_path, image_file_name))
        yolo_file_path = os.path.join(yolo_storage_path, image_file_path.stem + ".txt")
        
        if not os.path.exists(yolo_file_path): continue
        
        height, width, _ = cv2.imread(image_file_path).shape
        
        with open(yolo_file_path, "r") as file:
            for line in file:
                clas, cx, cy, w, h, conf = map(float, line.strip().split())
                cx *= width; cy *= height
                w *= width; h *= height

                coco_data.append({
                    "image_id": index, 
                    "category_id": int(clas), 
                    "bbox": [cx-w/2, cy-h/2, w, h], 
                    "score": conf
                })
    
    with open(coco_file_path, "w") as file:
        json.dump(coco_data, file, indent=4) 

def coco_eval(coco_gt_file_path, coco_dt_file_path):
    coco_gt = COCO(coco_gt_file_path)
    coco_dt = coco_gt.loadRes(coco_dt_file_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    image_storage_path = "C:/Users/hky3535/Desktop/train_toolkit/coco128/images/train2017"

    yolo2coco_gt(
        image_storage_path=image_storage_path, 
        yolo_storage_path="C:/Users/hky3535/Desktop/train_toolkit/coco128/labels/train2017", 
        coco_file_path="./coco_gt.json", 
        classes=coco_classes
    )
    
    yolo2coco_dt(
        image_storage_path=image_storage_path, 
        yolo_storage_path="C:/Users/hky3535/Desktop/train_toolkit/runs/detect/predict/labels", 
        coco_file_path="./coco_dt.json", 
        classes=coco_classes
    )

    coco_eval(
        coco_gt_file_path="./coco_gt.json", 
        coco_dt_file_path="./coco_dt.json"
    )




