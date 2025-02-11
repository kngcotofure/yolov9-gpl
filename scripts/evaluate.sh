CUDA_VISIBLE_DEVICES=1 python val_detr.py \
    --data data/org_coco.yaml \
    --img 640 \
    --batch 32 \
    --conf 0.001 \
    --iou 0.7 \
    --device 1 \
    --weights '/home/tiennv/nvtien/yoloxyz/weights/yolov9-c-rtdetr-tuning-1st/weights/best.pt' \
    --save-json \
    --name 'yolov9-deyo-val'