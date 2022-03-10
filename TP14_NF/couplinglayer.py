import torch
from utils import FlowModule, MLP


class CouplingLayer(FlowModule):
    def __init__(self, indim, hdim, dimcut=None, invx=False):
        super().__init__()
        self.indim = indim
        self.dimcut = dimcut
        self.hdim = hdim
        if dimcut is None:
            self.dimcut = self.indim // 2
        self.scale = MLP(self.dimcut, self.indim - self.dimcut, self.hdim)
        self.shift = MLP(self.dimcut, self.indim - self.dimcut, self.hdim)
        self.parity = invx

    def f(self, x):
        x0, x1 = x[:, :self.dimcut], x[:, self.dimcut:]
        if self.parity:
            x0, x1 = x1, x0
        s = self.scale(x0)
        t = self.shift(x0)
        z0 = x0
        z1 = torch.exp(s) * x1 + t  #
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def invf(self, z):
        z0, z1 = z[:, :self.dimcut], z[:, self.dimcut:]
        if self.parity:
            z0, z1 = z1, z0
        s = self.scale(z0)
        t = self.shift(z0)
        x0 = z0
        x1 = (z1 - t) * torch.exp(-s)
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det
