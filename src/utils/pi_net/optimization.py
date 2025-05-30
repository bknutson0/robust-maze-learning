"""From "Path independent equilibrium models ..." by Anil et. al."""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

##############################################################################################################
#
# Weight normalization. From MDEQ repository.
#
##############################################################################################################


def _norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return _norm(p.transpose(0, dim), 0).transpose(0, dim)


class WeightNorm:
    def __init__(self, names, dim):
        """Weight normalization module

        :param names: The list of weight names to apply weightnorm on
        :param dim: The dimension of the weights to be normalized
        """
        self.names = names
        self.dim = dim

    def compute_weight(self, module, name):
        g = getattr(module, name + '_g')
        v = getattr(module, name + '_v')
        return v * (g / _norm(v, self.dim))

    @staticmethod
    def apply(module, names, dim):
        fn = WeightNorm(names, dim)

        for name in names:
            weight = getattr(module, name)

            # remove w from parameter list
            del module._parameters[name]

            # add g and v as new parameters and express w as g/||v|| * v
            module.register_parameter(name + '_g', Parameter(_norm(weight, dim).data))
            module.register_parameter(name + '_v', Parameter(weight.data))
            setattr(module, name, fn.compute_weight(module, name))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module):
        for name in self.names:
            weight = self.compute_weight(module, name)
            delattr(module, name)
            del module._parameters[name + '_g']
            del module._parameters[name + '_v']
            module.register_parameter(name, Parameter(weight.data))

    def reset(self, module):
        for name in self.names:
            setattr(module, name, self.compute_weight(module, name))

    def __call__(self, module, inputs):
        # Typically, every time the module is called we need to recompute the weight. However,
        # in the case of TrellisNet, the same weight is shared across layers, and we can save
        # a lot of intermediate memory by just recomputing once (at the beginning of first call).
        pass


def weight_norm(module, names, dim=0):
    fn = WeightNorm.apply(module, names, dim)
    return module, fn


class VariationalHidDropout(nn.Module):
    def __init__(self, dropout=0.0, length_first=False):
        """Hidden-to-hidden (VD-based) dropout that applies the same mask at every time step and every layer
        :param dropout: The dropout rate (0 means no dropout is applied)
        :param temporal: Whether the dropout mask is the same across the temporal dimension (or only the depth dimension)
        """
        super(VariationalHidDropout, self).__init__()
        self.dropout = dropout
        self.mask = None
        self.length_first = length_first

    def reset_mask(self, bsz, d, length):
        if self.length_first:
            # Dimension (N, L, C)
            m = torch.zeros(bsz, 1, d).bernoulli_(1 - self.dropout)
        else:
            # Dimension (N, C, L)
            m = torch.zeros(bsz, d, 1).bernoulli_(1 - self.dropout)
        mask = m.requires_grad_(False) / (1 - self.dropout)
        self.mask = mask
        return mask

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        assert self.mask is not None, f'You need to reset mask before using {self.__class__.__name__}'
        mask = self.mask.expand_as(x)  # Make sure the dimension matches
        return mask * x


class VariationalHidDropout2d(VariationalHidDropout):
    def __init__(self, dropout=0.0, spatial=True):
        """Hidden-to-hidden (VD-based) 2D dropout that applies the same mask at every layer
        :param spatial: If True, then all pixels of the HxW feature map will be applied the
                        same mask as well (i.e., certain entire channels of all pixels may be
                        masked out).
        """
        super(VariationalHidDropout2d, self).__init__(dropout)
        self.spatial = spatial

    def reset_mask(self, bsz, d, H, W):
        # Dimension (N, C, H, W)
        if self.spatial:
            m = torch.zeros(bsz, d, 1, 1).bernoulli_(1 - self.dropout)
        else:
            m = torch.zeros(bsz, d, H, W).bernoulli_(1 - self.dropout)
        mask = m.requires_grad_(False) / (1 - self.dropout)
        self.mask = mask
        return mask
