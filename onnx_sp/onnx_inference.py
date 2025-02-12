import os
import yaml
import cv2
import time
import numpy as np
import torch
import onnxruntime as ort

from pathlib import Path


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

class YoloV9Deyo:
    def __init__(self, model_path, yaml_path=None):
        self.model_path = model_path
        self.load_model(model_path)

        if yaml_path is not None:
            with open(yaml_path, 'r') as file:
                prime_service = yaml.safe_load(file)
                self.class_names = prime_service['names']

    def load_model(self, model_path):
        self.model = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.inp_name = [x.name for x in self.model.get_inputs()]
        self.opt_name = [x.name for x in self.model.get_outputs()]
        _, _, h, w = self.model.get_inputs()[0].shape
        self.model_inpsize = (w, h)
        # print("Provider: ", self.model.get_providers())

    def inference(self, img, fp, save = False, save_path = Path("predict.jpg")):
        tensor, ratio, dwdh = self.preprocess(img, new_shape=self.model_inpsize, fp = fp)
        tensor = np.expand_dims(tensor, axis=0)
        # model prediction
        s0 = time.time()
        outputs = self.model.run(self.opt_name, dict(zip(self.inp_name, tensor)))[0]
        s1 = time.time()

        predictions = self.postprocess(preds=outputs, img=tensor, org_img=img, ratio=ratio, dwdh=dwdh, conf=0.4)

        if save:
            self.draw_predictions(predictions, img)
            cv2.imwrite(save_path, img)

        print("Time: ", round(s1-s0, 3))

    def preprocess(self, im:np.array, fp, new_shape=(640, 640), color=(114, 114, 114), scaleup=True) -> list:
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color
                                )  # add border

        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        if (fp == 32):
            im = np.ascontiguousarray(im, dtype=np.float32)
        elif (fp == 16):
            im = np.ascontiguousarray(im, dtype=np.float16)  # half precision float16
        else:
            print("Error fp")
            exit()
        im /= 255

        return im, r, (dw, dh)

    def postprocess(self, preds, img, org_img, ratio, dwdh, conf=0.4):
        if "deyo" in self.model_path.split('/')[-1] or 'rtdetr' in self.model_path.split('/')[-1]:
            return self._deyo_postprocess(preds, img, org_img, conf)

        return self._yolo_postprocess(preds, org_img, ratio, dwdh, conf)

    def _deyo_postprocess(self, preds, img, orig_img, conf = 0.4):
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds)
        nd = preds.shape[-1]
        bboxes, scores = preds.split((4, nd - 4), dim=-1)

        results = []
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = xywh2xyxy(bbox)
            bbox = scale_boxes(img.shape[3:], bbox, orig_img.shape)
            score, cls = scores[i].max(-1, keepdim=True)  # (300, 1)
            idx = score.squeeze(-1) > conf  # (300, )
            pred = torch.cat([bbox, score, cls], dim=-1)[idx]
            results.extend(pred.cpu().numpy())

        return results

    def _yolo_postprocess(self, pred, image, ratio, dwdh, det_thres = 0.5):
        if isinstance(pred, list):
            pred = np.array(pred)

        pred = pred[pred[:, 6] > det_thres] # get sample higher than threshold

        padding = dwdh*2
        bboxes, score, cls  = pred[:,1:5], pred[:,6], pred[:, 5]
        bboxes = ((bboxes[:, 0::] - np.array(padding)) / ratio).round()

        self.clip_coords(bboxes, image.shape)

        result = []
        for _box, _score, _cls in zip(bboxes, score, cls):
            result.append(np.concatenate((_box, _score, _cls), axis=None))

        return result

    def clip_coords(self, boxes, img_shape):
          # Clip bounding xyxy bounding boxes to image shape (height, width)
          boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
          boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
          boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
          boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2

    def draw_predictions(self, predictions, orig_img, thickness=2,
                         font_scale=0.5, font_color=(0, 255, 0), font_thickness=1):
        for line in predictions:
            x_min, y_min, x_max, y_max, score, cls_id = line[:6]
            score = round(score, 2)

            x_min, y_min, x_max, y_max, cls_id = map(int, [x_min, y_min, x_max, y_max, cls_id])
            label = f"{self.class_names.get(cls_id, 'Unknown')}: {score:.2f}"


            cv2.rectangle(orig_img, (x_min, y_min), (x_max, y_max), color=colors(int(cls_id), True), thickness=thickness)
            cv2.putText(orig_img, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors(int(cls_id), True), font_thickness)

        return orig_img

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()  # create instance for 'from utils.plots import colors'


if __name__ == '__main__':
    coco_path = '/home/tiennv/nvtien/datasets/stable_diffusion/coco/images/train'
    coco_images = [
        '000000033177.jpg',
        '000000081406.jpg',
        '000000129707.jpg',
        '000000178685.jpg',
        '000000227012.jpg',
        '000000276482.jpg',
        '000000324261.jpg',
        '000000371873.jpg',
        '000000420548.jpg',
        '000000469488.jpg',
        '000000518592.jpg',
        '000000567145.jpg',
    ]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    yolov9_deyo = YoloV9Deyo("weights/OJT_yolo/yolov9-c-rtdetr-tuning-1st/weights/yolov9_deyo_last.onnx",
                            '/home/tiennv/nvtien/projects/yolo-series/yolov9/data/coco.yaml')
   
    fp = 32

    for line in coco_images:
        # read image
        if os.path.isfile(os.path.join(coco_path, line)):
            image = cv2.imread(os.path.join(coco_path, line))
            name = line.split('.')[0]

            save_path = "output/" + name +  "_yolov9_deyo.jpg"
            result = yolov9_deyo.inference(image.copy(), fp, save=True, save_path= save_path)