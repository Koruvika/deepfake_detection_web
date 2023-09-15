import cv2
import numpy as np
import onnxruntime
import torch
from math import ceil
from itertools import product

from model.Settings import Settings


def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    return py_cpu_nms(dets, thresh)


class PriorBox:
    def __init__(self, settings: Settings, image_size=None):
        self.settings = settings
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in
                             self.settings.configs["steps"]]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.settings.configs["min_sizes"][k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x * self.settings.configs["steps"][k] / self.image_size[1] for x in
                                    [j + 0, j + 0.25, j + 0.5, j + 0.75]]
                        dense_cy = [y * self.settings.configs["steps"][k] / self.image_size[0] for y in
                                    [i + 0, i + 0.25, i + 0.5, i + 0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x * self.settings.configs["steps"][k] / self.image_size[1] for x in [j + 0, j + 0.5]]
                        dense_cy = [y * self.settings.configs["steps"][k] / self.image_size[0] for y in [i + 0, i + 0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.settings.configs["steps"][k] / self.image_size[1]
                        cy = (i + 0.5) * self.settings.configs["steps"][k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.settings.configs["clip"]:
            output.clamp_(max=1, min=0)
        return output


class FaceBoxes_ONNX:
    def __init__(self):
        self.settings = Settings("./assets/configs/face_detection.yaml")
        self.session = onnxruntime.InferenceSession(self.settings.configs["onnx_path"],
                                                    None,
                                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def __call__(self, img_):
        img_raw = img_.copy()
        img_raw = np.array(img_)
        # scaling to speed up
        scale = 1
        if self.settings.configs["scale_flag"]:
            h, w = img_raw.shape[:2]
            if h > self.settings.configs["HEIGHT"]:
                scale = self.settings.configs["HEIGHT"] / h
            if w * scale > self.settings.configs["WIDTH"]:
                scale *= self.settings.configs["WIDTH"] / (w * scale)
            if scale == 1:
                img_raw_scale = img_raw
            else:
                h_s = int(scale * h)
                w_s = int(scale * w)
                img_raw_scale = cv2.resize(img_raw, dsize=(w_s, h_s))
            img = np.float32(img_raw_scale)
        else:
            img = np.float32(img_raw)

        # forward
        im_height, im_width, _ = img.shape
        scale_bbox = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, ...]
        out = self.session.run(None, {'input': img})
        loc, conf = out[0], out[1]
        loc = torch.from_numpy(loc)
        prior_box = PriorBox(self.settings, (im_height, im_width))
        priors = prior_box.forward()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.settings.configs["variance"])
        if self.settings.configs["scale_flag"]:
            boxes = boxes * scale_bbox / scale / self.settings.configs["resize"]
        else:
            boxes = boxes * scale_bbox / self.settings.configs["resize"]

        boxes = boxes.cpu().numpy()
        scores = conf[0][:, 1]

        # ignore low scores
        inds = np.where(scores > self.settings.configs["confidence_threshold"])[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.settings.configs["top_k"]]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, self.settings.configs["nms_threshold"])
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:self.settings.configs["keep_top_k"], :]

        det_bboxes = []
        for b in dets:
            if b[4] > self.settings.configs["vis_threshold"]:
                xmin, ymin, xmax, ymax, score = b[0], b[1], b[2], b[3], b[4]
                bbox = [xmin, ymin, xmax, ymax, score]
                det_bboxes.append(bbox)

        return det_bboxes


    def upsize(self, dets, img_):
        img = np.array(img_)
        for bbox in dets:
            if bbox[4] < self.settings.configs["vis_threshold"]:
                continue
            x0 = max(bbox[0] - int((bbox[2] - bbox[0]) * self.settings.configs["bbox_scale"]), 0)
            x1 = min(bbox[2] + int((bbox[2] - bbox[0]) * self.settings.configs["bbox_scale"]), img.shape[1])
            y0 = max(bbox[1] - int((bbox[3] - bbox[1]) * self.settings.configs["bbox_scale"]), 0)
            y1 = min(bbox[3] + int((bbox[3] - bbox[1]) * self.settings.configs["bbox_scale"]), img.shape[0])
            bbox[0] = x0
            bbox[1] = y0
            bbox[2] = x1
            bbox[3] = y1
        return dets

    def viz_bbox(self, img_, dets):
        # show
        img = np.array(img_)
        for b in dets:
            if b[4] < self.settings.configs["vis_threshold"]:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        return img