# python export.py \
#   --weight 'yolov9-c-converted.pt' \
#   --conf-thres 0.5 \
#   --nms \
#   --max-wh 640 \
#   --simplify \
#   --include 'onnx_end2end'

# export Yolov9Pose
python export.py \
    --weights './weights/best_striped.pt' \
    --simplify \
    --topk-all 300 \
    --kpt-label 5 \
    --max-wh 640 \
    --include 'onnx_end2end'