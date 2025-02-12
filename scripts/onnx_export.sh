# python export.py \
#   --weight 'yolov9-c-converted.pt' \
#   --conf-thres 0.5 \
#   --nms \
#   --max-wh 640 \
#   --simplify \
#   --include 'onnx_end2end'

python export.py \
  --weight 'weights/OJT_yolo/yolov9-c-rtdetr-tuning-1st/weights/last.pt' \
  --imgsz 640 \
  --conf-thres 0.25 \
  --max-wh 640 \
  --simplify \
  --include 'onnx' \
  --detr \
  --opset 16 \
  --dynamic