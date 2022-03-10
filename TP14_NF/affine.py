from utils import FlowModule, check
import torch
from typing import Tuple
import torch.nn as nn
 
class Affine(FlowModule):
    def __init__(self, d) -> None:
        super().__init__()
        self.s = nn.Parameter(torch.ones(d, requires_grad=True))
        self.t = nn.Parameter(torch.zeros(d, requires_grad=True))

    def f(self, x):
        fx = torch.exp(self.s) * x + self.t
        detJf = (self.s).sum()
        return fx, torch.abs(detJf)

    def invf(self, y):
        finvx = (y - self.t) * torch.exp(- self.s)
        detinvJf = -(self.s).sum()
        return finvx, detinvJf


if __name__ == '__main__':
    d = 3
    x = torch.randn(d)
    flow = Affine(d)
    print('test f and finv : ', flow.f(x), flow.invf(x))
    print('compare x and f(f-1(x)) : ', x, flow.f(flow.invf(x)[0])[0])
    check(flow, x, 1e-6)