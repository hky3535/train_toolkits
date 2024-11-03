from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.predict(
    source="C:/Users/hky3535/Desktop/train_toolkit/coco128/images/train2017", 
    conf=0.25, 
    iou=0.7, 
    imgsz=640, 
    half=False, 
    device=None, 
    max_det=300, 
    vid_stride=1, 
    stream_buffer=False, 
    visualize=False, 
    augment=False, 
    agnostic_nms=False, 
    classes=None, 
    retina_masks=False, 
    embed=None, 
    project=None, 
    name=None, 
    show=False, 
    save=False, 
    save_frames=False, 
    save_txt=True, 
    save_conf=True, 
    save_crop=False, 
    show_labels=True, 
    show_conf=True, 
    show_boxes=True, 
    line_width=None, 
)


