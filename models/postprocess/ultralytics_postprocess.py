import sys
import math
import time
import cv2
import numpy as np
import torch
import torchvision
from .base_postprocess import Postprocess

class Ultralytics_Postprocess(Postprocess):

    def __init__(self):
        pass

    def postprocess(self):
        pass

    def load_class_names(self, dataset='COCO'):
        if dataset=='COCO':
            namesfile = './config/coco.names'
        elif dataset=='CROP':
            namesfile='./config/crop.names'
        else:
            print('[ERROR] No valid dataset was provided. Exiting!')
            sys.exit(0)
        class_names = []
        with open(namesfile, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        return class_names

    def extract_boxes(self, prediction, conf_thres=0.4, iou_thres=0.45, classes=None, agnostic=False,
                             multi_label=False,
                             labels=(), max_det=300):
        """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        boxes = self.deserialize_bytes_float(prediction.raw_output_contents[0])
        boxes = np.reshape(boxes, prediction.outputs[0].shape)
        prediction = boxes

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
        if np.any(np.array(xc)) == True:
            # Checks
            assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
            assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

            # Settings
            min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
            max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
            time_limit = 10.0  # seconds to quit after
            redundant = True  # require redundant detections
            multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
            merge = False  # use merge-NMS
            t = time.time()
            output = [torch.zeros((0, 6))] * prediction.shape[0]
            for xi, x in enumerate(prediction):  # image index, image inference
                # Apply constraints
                # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
                x = x[xc[xi]]  # confidence

                # Cat apriori labels if autolabelling
                if labels and len(labels[xi]):
                    l = labels[xi]
                    v = torch.zeros((len(l), nc + 5), device=x.device)
                    v[:, :4] = l[:, 1:5]  # box
                    v[:, 4] = 1.0  # conf
                    v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                    x = torch.cat((x, v), 0)

                # If none remain process next image
                if not x.shape[0]:
                    continue

                # Compute conf
                x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

                # Box (center x, center y, width, height) to (x1, y1, x2, y2)
                box = self.xywh2xyxy(x[:, :4]).astype(np.float32)

                # Detections matrix nx6 (xyxy, conf, cls)
                if multi_label:
                    i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                    x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
                else:  # best class only
                    x = torch.tensor(x.astype(np.float64))
                    conf, j = x[:, 5:].max(1, keepdim=True)
                    x = torch.cat((torch.from_numpy(box), conf, j.float()), 1)[conf.view(-1) > conf_thres]

                # Filter by class
                if classes is not None:
                    x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

                # Apply finite constraint
                # if not torch.isfinite(x).all():
                #     x = x[torch.isfinite(x).all(1)]

                # Check shape
                n = x.shape[0]  # number of boxes
                if not n:  # no boxes
                    continue
                elif n > max_nms:  # excess boxes
                    x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

                # Batched NMS
                c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
                boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
                i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
                if i.shape[0] > max_det:  # limit detections
                    i = i[:max_det]
                if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                    # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                    iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                    weights = iou * scores[None]  # box weights
                    x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                    if redundant:
                        i = i[iou.sum(1) > 1]  # require redundancy

                output[xi] = x[i]
                if (time.time() - t) > time_limit:
                    print(f'WARNING: NMS time limit {time_limit}s exceeded')
                    break  # time limit exceeded
            output = output[0].cpu().numpy()
            output = [output[:, 0:4], output[:, 5], output[:, 4]]
        else:
            # No bounding boxes found within the given confidence and IoU range
            output = [[] for i in range(3)]
        return output

    def plot_boxes_cv2(self,img, boxes, savename=None, class_names=None, color=None):
        img = np.copy(img)
        colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

        def get_color(c, x, max_val):
            ratio = float(x) / max_val * 5
            i = int(math.floor(ratio))
            j = int(math.ceil(ratio))
            ratio = ratio - i
            r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
            return int(r * 255)

        width = img.shape[1]
        height = img.shape[0]
        for i in range(len(boxes)):
            box = boxes[i]
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)

            if color:
                rgb = color
            else:
                rgb = (255, 0, 0)
            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                cls_id = box[6]
                print('%s: %f' % (class_names[cls_id], cls_conf))
                classes = len(class_names)
                offset = cls_id * 123457 % classes
                red = get_color(2, offset, classes)
                green = get_color(1, offset, classes)
                blue = get_color(0, offset, classes)
                if color is None:
                    rgb = (red, green, blue)
                img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
        if savename:
            print("save plot results to %s" % savename)
            cv2.imwrite(savename, img)
        return img