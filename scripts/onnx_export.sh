python export.py \
  --weight 'yolov9-c-converted.pt' \
  --conf-thres 0.5 \
  --nms \
  --max-wh 640 \
  --simplify \
  --include 'onnx_end2end'