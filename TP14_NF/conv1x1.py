import torch
from utils import FlowModule


class conv1x1(FlowModule):
    def __init__(self, indim, outdim):
        super().__init__()
        self.indim = indim
        self.outdim = outdim

        weight = torch.randn(self.indim, self.outdim)
        weight = torch.nn.init.orthogonal_(weight)
        d = torch.lu(weight)
        w_p, w_l, w_u = torch.lu_unpack(d[0], d[1])
        w_s = torch.diag(w_u)
        w_u = torch.triu(w_u, 1)
        u_mask = torch.triu(torch.ones_like(w_u), 1)
        l_mask = u_mask.T

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", u_mask)
        self.register_buffer("l_mask", l_mask)
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = torch.nn.Parameter(w_l)
        self.w_s = torch.nn.Parameter(torch.log(torch.abs((w_s))))
        self.w_u = torch.nn.Parameter(w_u)

    def _get_weight(self):
        weight = self.w_p @ (self.w_l * self.l_mask + self.l_eye) @ (
            (self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        return weight

    def f(self, x):
        y = x @ self._get_weight()
        log_det = self.w_s.sum()
        return y, log_det

    def invf(self, x):
        y = x @ self._get_weight().inverse()
        log_det = -self.w_s.sum()
        return y, log_det
