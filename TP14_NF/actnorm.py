from utils import FlowModule, check
import torch.nn as nn
import torch
 
class actNorm(FlowModule):
    def __init__(self, d) -> None:
        super().__init__()
        self.s = torch.randn(d)
        self.t = torch.randn(d)
        self.norm=False

    def f(self, x):
        if not self.norm:
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.norm = True
        fx = torch.exp(self.s) * x + self.t
        detJf = (self.s).sum()
        return fx, torch.abs(detJf)

    def invf(self, y):
        if not self.norm:
            self.s.data = (-torch.log(y.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(y * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.norm = True
        finvx = (y - self.t) * torch.exp(- self.s)
        detinvJf = (- self.s).sum()
        return finvx, detinvJf
