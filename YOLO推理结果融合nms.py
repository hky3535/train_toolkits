import numpy
import cv2
import os
from pathlib import Path

def yolo2list(yolo_file_path, image_file_path):
    height, width, _ = cv2.imread(image_file_path).shape
    results = list()
    with open(yolo_file_path, "r") as file:
        for line in file:
            clas, cx, cy, w, h, conf = map(float, line.strip().split())
            cx *= width; cy *= height
            w *= width; h *= height
            results.append([int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2), float(conf), int(clas)])
    return results

def list2yolo(yolo_list, image_file_path):
    height, width, _ = cv2.imread(image_file_path).shape
    lines = str()
    for r in yolo_list:
        x0, y0, x1, y1, conf, clas = map(float, r)
        w, h = x1-x0, y1-y0
        cx, cy = x0+w/2, y0+h/2
        cx /= width; cy /= height
        w /= width; h /= height
        lines += f"{int(clas)} {cx} {cy} {w} {h} {conf}\n"
    return lines
    
def yolo_merge_nms(image_storage_path, yolo_storage_paths, yolo_merge_storage_path):
    for image_file_name in os.listdir(image_storage_path): # 遍历每张图片
        image_file_path = Path(os.path.join(image_storage_path, image_file_name))
        results = list()
        for yolo_storage_path in yolo_storage_paths: # 遍历所有结果
            yolo_file_path = os.path.join(yolo_storage_path, image_file_path.stem + ".txt")
            if not os.path.exists(yolo_file_path): continue
            results += yolo2list(yolo_file_path=yolo_file_path, image_file_path=image_file_path)
        # 至此得到当前图片的所有模型的推理结果
        # 对每个类别分别进行nms
        results_final = list()
        for clas in set([r[5] for r in results]):
            results_part = [r for r in results if r[5] == clas]
            indices = cv2.dnn.NMSBoxes(
                bboxes=[r[:4] for r in results_part], 
                scores=[r[4] for r in results_part], 
                score_threshold=0.0, nms_threshold=0.5, 
                eta=1, top_k=0
            )
            results_final += [results_part[i] for i in indices]
        # 至此得到当前图片的所有模型的推理结果的融合结果
        lines = list2yolo(yolo_list=results_final, image_file_path=image_file_path)
        with open(os.path.join(yolo_merge_storage_path, image_file_path.stem + ".txt"), "w") as file:
            file.write(lines)


if __name__ == "__main__":
    image_storage_path = "C:/Users/hky3535/Desktop/train_toolkit/coco128/images/train2017"
    yolo_storage_paths = [
        "C:/Users/hky3535/Desktop/train_toolkit/runs/detect/predict/labels", 
        "C:/Users/hky3535/Desktop/train_toolkit/runs/detect/predict/labels"
    ]
    yolo_merge_nms(
        image_storage_path=image_storage_path, 
        yolo_storage_paths=yolo_storage_paths, 
        yolo_merge_storage_path="C:/Users/hky3535/Desktop/train_toolkit/merge"
    )
