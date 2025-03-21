import torch as th
import torchvision.transforms.functional as F
import torchvision.utils as tu
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

from .dct import dct_2d, idct_2d


class EMAHelper:
    # Simplified from https://github.com/ermongroup/ddim/blob/main/models/ema.py:
    def __init__(self, module, mu=0.999, device=None):
        self.module = module
        self.mu = mu
        self.device = device
        self.shadow = {}
        # Register:
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self):
        locs = self.module.locals
        module_copy = type(self.module)(*locs).to(self.device)
        module_copy.load_state_dict(self.module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def a_(s, t, a_k):
    #  t > s
    return th.exp(-a_k * (t - s))


def a_2(s, t, a_k):
    #  t > s
    return th.exp(-2 * a_k * (t - s))


def v_(s, t, a_k):
    # t > s
    coeff = 1 / (2 * a_k)
    return coeff * (1 - a_2(s, t, a_k))


def euler_discretization(x, xp, nn, energy, device="cuda"):
    # Assumes x has shape [T, B, C, H, W].
    # Assumes x[0] already initialized.
    # We normalize by D = C * H * W the drift squared norm, and not by scalar sigma.
    # Fills x[1] to x[T] and xp[0] to xp[T - 1].
    T = x.shape[0] - 1  # Discretization steps.
    B = x.shape[1]
    dt = th.full(size=(x.shape[1],), fill_value=1.0 / T, device=device)
    drift_norms = 0.0

    _, b, c, h, w = x.shape

    freqs = np.pi * th.linspace(0, h - 1, h) / h
    freq = (freqs[:, None] ** 2 + freqs[None, :] ** 2).to(x.device)
    frequencies_squared = freq + 1.0
    a_k = frequencies_squared[None, None]
    sigma_k = th.pow(a_k, -0.01) / energy

    for i in range(1, T + 1):
        t = dt * (i - 1)
        alpha_t = nn(x[i - 1], t)
        drift_norms = drift_norms + th.mean(alpha_t.reshape(B, -1) ** 2, dim=1)

        alpha_t = dct_2d(alpha_t, norm="ortho")
        x_i = dct_2d(x[i - 1], norm="ortho")
        t_ = t[:, None, None, None]
        t_end = th.ones_like(t_)
        xp_coeff = alpha_t
        xp[i - 1] = idct_2d(xp_coeff, norm="ortho")
        control = (a_(t_, t_end, a_k) * alpha_t - a_2(t_, t_end, a_k) * x_i) / v_(
            t_, t_end, a_k
        )
        drift_t = (-a_k * x_i + control) * dt[:, None, None, None]
        eps_t = dct_2d(th.randn_like(x[i - 1]), norm="ortho")

        if i == T:
            diffusion_t = 0
        else:
            diffusion_t = sigma_k * th.sqrt(dt[:, None, None, None]) * eps_t
        x[i] = idct_2d(x_i + drift_t + diffusion_t, norm="ortho")

    drift_norms = drift_norms / T
    return drift_norms.cpu()


def load_state_dict(checkpoint, saves, ckpt_keys):
    for i, k in enumerate(ckpt_keys):
        state_dict = checkpoint[k]
        new_state_dict = OrderedDict()
        for cc, v in state_dict.items():
            name = cc.replace("module.", "")  # remove `module.`
            new_state_dict[name] = v
        model_state_dict = saves[i].state_dict()
        filtered_state_dict = {
            cc: v for cc, v in new_state_dict.items() if cc in model_state_dict
        }
        model_state_dict.update(filtered_state_dict)
        saves[i].load_state_dict(model_state_dict)


def draw_square(img, x1, y1, x2, y2, thick=1, color="r"):
    img = img.clone()
    if color == "r":
        rgb = (1, 0, 0)
    elif color == "b":
        rgb = (0, 0, 1)
    else:
        rgb = (1, 1, 1)
    img[y1 : y1 + thick, x1:x2, 0] = rgb[0]
    img[y2 - thick : y2, x1:x2, 0] = rgb[0]
    img[y1:y2, x1 : x1 + thick, 0] = rgb[0]
    img[y1:y2, x2 - thick : x2, 0] = rgb[0]

    img[y1 : y1 + thick, x1:x2, 1] = rgb[1]
    img[y2 - thick : y2, x1:x2, 1] = rgb[1]
    img[y1:y2, x1 : x1 + thick, 1] = rgb[1]
    img[y1:y2, x2 - thick : x2, 1] = rgb[1]

    img[y1 : y1 + thick, x1:x2, 2] = rgb[2]
    img[y2 - thick : y2, x1:x2, 2] = rgb[2]
    img[y1:y2, x1 : x1 + thick, 2] = rgb[2]
    img[y1:y2, x2 - thick : x2, 2] = rgb[2]
    return img


def make_data_grid(res):
    gridx = th.tensor(np.linspace(0, 1, res), dtype=th.float32)
    gridx = gridx.reshape(1, res, 1, 1).repeat([1, 1, res, 1])
    gridy = th.tensor(np.linspace(0, 1, res), dtype=th.float32)
    gridy = gridy.reshape(1, 1, res, 1).repeat([1, res, 1, 1])
    grid = th.cat((gridx, gridy), dim=-1).reshape(1, -1, 2)
    return grid
