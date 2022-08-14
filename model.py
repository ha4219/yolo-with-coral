import os

import pycoral.utils.edgetpu as etpu
from pycoral.adapters import common


class Model:
    def __init__(self, model_file, names_file, conf_thresh=0.25, iou_thresh=0.45, filter_classes=None, agnostic_nms=False, max_det=1000) -> None:
        model_file = os.path.abspath(model_file)


    
    def forward(self, x):
        return x
    

