python train.py \
    --weights 'weights/gelan-c.pt' \
    --cfg 'models/detect/gelan-keypoint.yaml' \
    --data 'datahub/face-gender-dataset-v2-with-landmark/data.yaml' \
    --hyp 'data/hyps/hyp.scratch-high.yaml' \
    --batch-size 12 \
    --name "landmark-Ikeypoint" \
    --kpt-label 5 \
    --epochs 150 \
    --device '0'