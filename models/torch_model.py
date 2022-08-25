'''torch model implementation'''
import torch


class Model:
    '''
        Torch Model
    '''
    def __init__(self, model_path: str = 'models/torch/640/s.cpu.torchscript') -> None:
        self.model_path = model_path
        self.model = torch.jit.load(model_path)

    def forward(self, x):
        res = self.model(x)
        return res


x = torch.randn((1, 3, 640, 640))
m = Model()

print(m.forward(x)[0].shape)