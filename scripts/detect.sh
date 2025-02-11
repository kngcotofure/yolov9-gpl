CUDA_VISIBLE_DEVICES=1 python detect_detr.py \
    --source data/images \
    --img 640 \
    --conf-thres 0.38 \
    --device 1 \
    --weights './yolov9-c-rtdetr-tuning-1st/weights/best.pt' \
    --name yolov9_deyo_640_detect