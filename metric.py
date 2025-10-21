import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.metrics import box_iou, ap_per_class
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, TQDM_BAR_FORMAT, LOGGER, scale_boxes, xywh2xyxy


def img2predict_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}predicts{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def detect_box_format(predn):
    boxes = predn[:, :4]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    max_val = boxes.max().item()
    is_normalized = max_val <= 1.0
    xyxy_like = ((x2 > x1) & (y2 > y1)).float().mean().item() > 0.8
    fmt = 'xyxy' if xyxy_like else 'xywh'

    return f"{fmt}-{'normalized' if is_normalized else 'pixel'}"


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def main(
    data,
    imgsz=(640, 640),  # inference size (height, width)
    batch_size:int = 1,
    stride: int = 32,
    device: str ='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    single_cls:bool = False,
    pad: float = 0.5,
    rect: bool = False,
    workers: int = 4,
    min_items: int = 0,
    save_dir = "benchmark",
    verbose= False,
    training= False,
):
    save_dir = Path(save_dir)
    data = check_dataset(data)  # check    
    dataloader = create_dataloader(data["val"],
                                       imgsz[0],
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       min_items=min_items,
                                       )[0]
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data['names']) != 1 else data['names']  # class names
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    seen = 0
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    jdict, stats, ap, ap_class = [], [], [], []

    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # load predict
        predict_path = img2predict_paths(paths)[0]
        if os.path.isfile(predict_path):
            with open(predict_path, 'r') as f:
                pd = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        else: # label missing
            print("File not found: ", predict_path)
            pd = np.zeros((0, 6), dtype=np.float32)
        pd = torch.from_numpy(pd).to(device)
        preds = [pd]
        
        nb, _, height, width = im.shape  # batch size, channels, height, width
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1
            
            # Predictions
            if single_cls:
                pred[:, 5] = 0
            if isinstance(pred, torch.Tensor):
                predn = pred.clone()
            else:
                predn = pred.copy()  # or pred.copy() for NumPy arrays
            
            fmt = detect_box_format(predn)
            if fmt == 'xywh-normalized':
                predn[:, :4] *= torch.tensor([width, height, width, height], device=predn.device)
                predn[:, :4] = xywh2xyxy(predn[:, :4])
                scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space labels
            elif fmt == 'xywh-pixel':
                predn[:, :4] = xywh2xyxy(predn[:, :4])
            elif fmt == 'xyxy-normalized':
                predn[:, :4] *= torch.tensor([width, height, width, height], device=predn.device)
            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, pconf, pcls, tcls)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in val set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size inference')
    parser.add_argument('--stride', type=int, default=32, help='Model stride')
    parser.add_argument('--single-cls', action='store_true', help='Group the dataset into a single class')
    parser.add_argument('--pad', type=float, default=0.5, help='Setup padding for inference')
    parser.add_argument('--rect', action='store_true', help='Setup rectangular inference')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--min-items', type=int, default=0, help='Minimum number of items in dataset')
    parser.add_argument('--save-dir', type=str, default='benchmark', help='Directory to save results')
    parser.add_argument('--verbose', action='store_true', help='Print detailed results')
    parser.add_argument('--training', action='store_true', help='Training mode, used for debugging')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print(vars(opt))
    
    main(**vars(opt))