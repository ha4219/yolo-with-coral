import numpy as np
import torch
from glob import glob
import cv2
from pathlib import Path

# from models.tf_model import Model as TFModel
from models.torch_model import Model as TorchModel
from nms import non_max_suppression
from util import get_image_tensor, box_iou, scale_coords, xywh2xyxy, ap_per_class, Annotator, xywhn2xyxy, xyxy2xywh



CONF_THRES = 0.01
def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = np.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return np.array(correct, dtype=bool)

nc = 1

stats = []

iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
niou = iouv.size

m = TorchModel('models/torch/640/s.cpu.v7.torchscript')
images = sorted(glob('sample/images/*'))[:10]
labels = sorted(glob('sample/labels/*'))[:10]
y = []
for label in labels:
    with open(label, 'r') as f:
        lines = []
        for line in f.readlines():
            lines.append(list(map(float, line.strip().split())))
        if lines:
            lines = np.array(lines)
        else:
            lines = np.ndarray((0, 5))
        y.append(lines)
x = [(get_image_tensor(cv2.imread(img), 640),img) for img in images]
for idx, (((im, shapes, im0, cratio), paths), (label)) in enumerate(zip(x, y)):
    label_origin = label.copy()
    out = m.forward(torch.from_numpy(im).unsqueeze(0).float())
    print(out.shape)
    out = non_max_suppression(prediction=out, conf_thres=0.0001, iou_thres=0.6, labels=[], agnostic=True)
    _, height, width = im.shape
    im0 = cv2.imread(paths)
    annotator = Annotator(im0, line_width=10)
    padw, padh = shapes[-1][1]
    label[:, 1:] *= [width, height, width, height]
    for si, pred in enumerate(out):
        shape = shapes[0]
        nl, npr = label.shape[0], pred.shape[0]
        correct = np.zeros((npr, niou)).astype(bool)
        pred[:, 5] = 0
        predn = np.copy(pred)
        scale_coords(im.shape[1:], predn[:, :4], shape, shapes[1])  # native-space pred
        if npr == 0:
            if nl:
                stats.append((correct, *np.zeros((2, 0)), label[:, 0]))
            continue

        for *xyxy, conf, cls in reversed(predn):
            if conf < CONF_THRES: continue
            c = int(cls)
            xywh = xyxy2xywh(np.expand_dims(xyxy, axis=0))
            line = f'{round(conf, 2)}'
            annotator.box_label(xyxy, line, (0, 0, 255))

        if nl:
            tbox = xywh2xyxy(label[:, 1:5])
            scale_coords(im.shape[1:], tbox, shape, shapes[1])  # native-space labels

            labeln = np.concatenate((label[:, 0:1], tbox), 1)
            # for conf, *xyxy in labeln:
            #     print(xyxy)
            #     annotator.box_label(xyxy, None, (255, 0, 0))
            correct = process_batch(predn, labeln, iouv)
        stats.append((correct, pred[:, 4], pred[:, 5], label[:, 0]))  # (correct, conf, pcls, tcls)

        if nl:
            h, w = shape
            label_origin = xywh2xyxy(label_origin[:, 1:5])
            label_origin *= [w, h, w, h]
            for xyxy in label_origin:
                annotator.box_label(xyxy, None, (255, 0, 0))
        
        im0 = annotator.result()
        name = paths.split('/')[-1]
        cv2.imwrite(f'{name}', im0)

stats = [np.concatenate(x, 0) for x in zip(*stats)]
if len(stats):
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=True, save_dir=Path('res'),  names={0: 'Danger'})
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map95 = p.mean(), r.mean(), ap50.mean(), ap.mean()
nt = np.bincount(stats[3].astype(int), minlength=nc)
pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format

print(ap50, ap)

print('precision\t', mp)
print('recall\t', mr)
print('map .50\t', map50)
print('map .95\t', map95)