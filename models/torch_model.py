'''torch model implementation'''
import torch
import numpy as np


class Model:
    '''
        Torch Model
    '''
    def __init__(self, model_path: str = 'models/torch/640/s.cpu.torchscript') -> None:
        self.model_path = model_path
        self.model = torch.jit.load(model_path)
        self.is7 = 'v7' in model_path

    def forward(self, x):
        ''' forward '''
        res = self.model(x)
        if self.is7: return res[0].detach().numpy()
        return res[0].numpy()

