import torch
from models.DHN import Munkrs


if __name__ == '__main__':
    mod = Munkrs(1, 256, 1, True, 1, False, False)
    weights = torch.load('/home/wanghao/github/deepmot/model_weights/DHN.pth')
    mod.load_state_dict(weights)

    x = torch.rand(1, 4, 4)
    x = x - x * torch.ones_like(x)
    with torch.no_grad():
        y = mod(x)
        print(y)
