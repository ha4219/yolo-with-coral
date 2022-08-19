import torch


class Model:
    def __init__(self, model_path: str = 'models/torch/640/s.cpu.torchscript') -> None:
        self.model_path = model_path
        self.model = torch.jit.load(model_path)

