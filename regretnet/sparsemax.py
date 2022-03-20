"""Sparsemax activation function.
Pytorch implementation of Sparsemax function from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)
"""

from __future__ import division

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        number_of_logits = x.size(self.dim)

        # Translate input by max for numerical stability
        x = (x - torch.max(x, dim=self.dim, keepdim=True)[0])

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(x, dim=self.dim, descending=True)[0]
        #shaped (batch_size, # of classes, otherdim)
        view_shape = [1 if i!=self.dim else -1 for i in range(len(x.size()))]
        num_list = torch.arange(start=1, end=number_of_logits+1, device=x.device, dtype=torch.float32).view(*view_shape)

        # Determine sparsity of projection
        bound = 1 + num_list * zs
        cumulative_sum_zs = torch.cumsum(zs, self.dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).float()
        k = torch.max(is_gt * num_list, self.dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, self.dim, keepdim=True) - 1) / k

        # Sparsemax
        output = torch.max(torch.zeros_like(x), x - taus)
        self.output = output

        return output

class Sparsemax_Old(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax_Old, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = input.size()
        input = input.reshape(-1, input.size(self.dim))

        dim = 1
        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits+1, device=input.device, dtype=torch.float32).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        # self.output = torch.max(torch.zeros_like(input), input - taus)

        # output = self.output.view(original_size)

        return torch.max(torch.zeros_like(input), input - taus).view(original_size)

