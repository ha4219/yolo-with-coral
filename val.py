from glob import glob

import numpy as np

from models.tf_model import Model
from util import get_image_tensor

model = Model('models/320/s.tflite', 'data/tree.yaml', conf_thresh=0.25, iou_thresh=0.45)

images = glob('/data/images/val/*')
labels = glob('/data/labels/val/*')

x = [get_image_tensor(image, model.input_size[0]) for image in images]
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


preds = []
labels = []
for (full_image, net_image, pad), (label) in zip(x, y):
    pred = model.forward(net_image)
    preds.append(pred)
    labels.append(label)
    print('pred', pred)
    print('label', label)
