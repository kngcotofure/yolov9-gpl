import cv2
import numpy as np
import onnxruntime as ort
from typing import List


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

def draw_predictions(
    predictions,
    orig_img,
    thickness=2,
    font_scale=0.5,
    font_thickness=1,
    kpt_label=False,
):
    """
        Draw bounding boxes and optionally keypoints on an image.
        Args:
            predictions (List[List[float]]): List of predictions
            orig_img (np.ndarray): Original input image (BGR).
            thickness (int): Line thickness for bounding boxes.
            font_scale (float): Font scale for text labels.
            font_thickness (int): Thickness of text labels.
            kpt_label (bool): Whether to draw keypoints if available.

        Returns:
            np.ndarray: Image with drawn predictions.
    """
    for line in predictions:
        x_min, y_min, x_max, y_max, score, cls_id = line[:6]
        score = round(score, 2)
        cls_id = int(cls_id)

        x_min, y_min, x_max, y_max, cls_id = map(
            int, [x_min, y_min, x_max, y_max, cls_id]
        )
        label = f"{cls_id}: {score:.2f}"
        cv2.rectangle(
            orig_img,
            (x_min, y_min),
            (x_max, y_max),
            color=colors(cls_id, True),
            thickness=thickness,
        )
        cv2.putText(
            orig_img,
            label,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            colors(cls_id, True),
            font_thickness,
        )

        if kpt_label:
            kpts = line[6:]
            kpts = np.array(kpts).reshape(-1, 3)
            orig_img = draw_landmark(orig_img, kpts, 2)

    return orig_img


def draw_landmark(image, kpts, radius=3, threshold=0.5, colors=None):
    """
        Draw keypoints on an image.

        Args:
            image (np.ndarray): Image to draw on.
            kpts (np.ndarray): Keypoints in shape (N, 3) or flat list [x1, y1, v1, ...].
                Each keypoint has (x, y, visibility).
            radius (int): Radius of the drawn circle.
            threshold (float): Visibility threshold (v < threshold → skip point).
            colors (List[Tuple[int, int, int]]): Optional list of colors for each keypoint.

        Returns:s
            np.ndarray: Image with drawn keypoints.
    """
    if kpts.ndim == 2 and kpts.shape[1] == 3:
        pts = kpts
    else:
        k = len(kpts) // 3 if kpts.ndim == 1 else kpts.shape[1] // 3
        pts = kpts.reshape(-1, k, 3)[0]  # (K,3)

    K = pts.shape[0]

    if colors is None:
        if colors is None:
            colors = [
                (255, 0, 0),  # red
                (0, 255, 0),  # green
                (0, 0, 255),  # blue
                (255, 255, 0),  # yellow
                (255, 0, 255),  # pink
            ]
        colors = colors[:K]
    for i, (x, y, v) in enumerate(pts):
        if v < threshold:  # visibility
            continue
        color = colors[i % len(colors)]
        cv2.circle(image, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)

    return image


class ONNXIKeypoint:
    def __init__(self, model_path):
        self.model = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.inp_name = [x.name for x in self.model.get_inputs()]
        self.opt_name = [x.name for x in self.model.get_outputs()]
        _, _, h, w = self.model.get_inputs()[0].shape

        if isinstance(h, str) or isinstance(w, str):
            print(
                "[WARNING]: Model input shape must be static, not dynamic. Please check the ONNX model."
            )
            h, w = 640, 640
        self.model_inpsize = (w, h)

        print("Model initalize sucess with provider: ", self.model.get_providers())

    def inference(self, img, thresold:float=0.25) -> List[List[float]]:
        """
        Run inference on an image and return processed detections.

        Args:
            img (np.ndarray): Input image in BGR format.
            thresold (float): Confidence threshold for filtering detections.

        Returns:
            List[List[float]]: Each detection is a list of
                [x_min, y_min, x_max, y_max, score, class_id, ...keypoints].
        """
        tensor, ratio, dwdh = self.preprocess(img, new_shape=self.model_inpsize)
        _tensor = np.expand_dims(tensor, axis=0)

        # model prediction
        outputs = self.model.run(self.opt_name, dict(zip(self.inp_name, _tensor)))[0]

        predictions = self.postprocess(
            preds=outputs,
            image=img,
            ratio=ratio,
            dwdh=dwdh,
            det_thresh=thresold,
        )
        return predictions

    def preprocess(
        self, im: np.array, new_shape=(640, 640), color=(114, 114, 114), scaleup=True
    ) -> list:
        """
        Resize and pad an image to the model's expected input size.
        """ 
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
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border

        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im, dtype=np.float32)
        im /= 255

        return im, r, (dw, dh)

    def postprocess(
        self,
        preds: List[np.ndarray],
        image: np.ndarray,
        ratio,
        dwdh,
        det_thresh: float = 0.25,
    ) -> List[List[float]]:
        """
        Postprocess the predictions (bounding boxes, scores, classes and keypoint).

        Args:
            preds (np.ndarray): Raw model outputs.
            image (np.ndarray): Original image.
            ratio (float): Resize ratio used during preprocessing.
            dwdh (Tuple[float, float]): Padding offsets (dw, dh).
            det_thresh (float): Confidence threshold.
        Returns:
            List[List[float]]: Processed detections with bbox + score + cls + keypoints.
        """

        preds = preds[preds[:, 6] > det_thresh]  # get sample higher than threshold
        if len(preds) == 0:
            return []

        # padding = dwdh * 2
        bboxes, score, cls = preds[:, 1:5].astype(np.float32), preds[:, 6], preds[:, 5]
        kpts = preds[:, 7:] if preds.shape[1] > 7 else None

        padding = np.array([dwdh[0], dwdh[1], dwdh[0], dwdh[1]], dtype=np.float32)
        bboxes = ((bboxes[:, 0::] - padding) / ratio).round()
        self.clip_coords(bboxes, image.shape)

        num_kpts = kpts.shape[1] // 3
        kpts = kpts.reshape(-1, num_kpts, 3)
        for i in range(num_kpts):
            kpts[:, i, 0] = (kpts[:, i, 0] - dwdh[0]) / ratio  # x
            kpts[:, i, 1] = (kpts[:, i, 1] - dwdh[1]) / ratio  # y

        results = []
        for idx in range(len(bboxes)):
            if kpts is not None:
                results.append(np.concatenate((bboxes[idx], score[idx], cls[idx], kpts[idx]), axis=None))
            else:
                results.append(np.concatenate((bboxes[idx], score[idx], cls[idx]), axis=None))

        return results

    def clip_coords(self, coords: np.ndarray, img_shape, step: int = 2):
        coords[:, 0::step] = np.clip(coords[:, 0::step], 0, img_shape[1])
        coords[:, 1::step] = np.clip(coords[:, 1::step], 0, img_shape[0])


if __name__ == "__main__":
    import os
    import glob
    from tqdm import tqdm

    output_dir = "runs/onnx"
    os.makedirs(output_dir, exist_ok=True)
    model = ONNXIKeypoint("./best_striped-end2end.onnx")

    samples = glob.glob("data/images/*.jpg")
    for i in tqdm(range(len(samples)), desc="Predict ONNX"):
        img = cv2.imread(samples[i])
        result = model.inference(img, thresold=0.25)
        img = draw_predictions(result, img, kpt_label=True)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(samples[i])), img)
