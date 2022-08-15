import pickle
import numpy as np
from utils import xywh2xyxy, box_iou, ap_per_class
from pathlib import Path

nc = 1

with open('data/pred.npy', 'rb') as f:
    preds = pickle.load(f)
with open('data/label.npy', 'rb') as f:
    labels = pickle.load(f)

iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
niou = iouv.size

# names = ['Danger']
# names = dict(enumerate(model.names if hasattr(model, 'names') else model.module.names))

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
            matches = np.cat((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return np.array(correct, dtype=bool)

stats = []
for pred, label in zip(preds, labels):
    shape = [2160, 3840]
    pred = pred[0]
    nl, npr = label.shape[0], pred.shape[0]  # number of labels, predictions
    correct = np.zeros((npr, niou)).astype(bool)
    predn = np.copy(pred)

    if npr == 0:
        if nl:
            stats.append((correct, *np.zeros((2, 0)), label[:, 0]))
        continue
    
    if nl:
        tbox = xywh2xyxy(label[:, 1:])
        labeln = np.concatenate((label[:, 0:1], tbox), 1)
        # TODO cal padding coordinate
        correct = process_batch(predn, labeln, iouv)
    stats.append((correct, pred[:, 4], pred[:, 5], label[:, 0]))  # (correct, conf, pcls, tcls)

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