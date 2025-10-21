
python detect_kpts.py \
    --weights './weights/best_striped.pt' \
    --source 'data/images' \
    --img 640 \
    --device 'cpu' \
    --kpt-label 5 \
    --conf-thres 0.25