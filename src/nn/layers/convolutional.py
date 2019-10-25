from ..enums import LRPRule, ExplainingMethod
from ..utils import CovarianceCalculator

import torch
import torch.nn as nn
import numpy as np


class Convolutional(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activation_fn=False, lrp_rule=LRPRule.alpha_beta, data_mean=None, data_std=None):
        super(Convolutional, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.lrp_rule = lrp_rule
        self.activation_fn = activation_fn

        self.X = 0
        self.pre_activation = 0
        self.alpha = 1.0
        self.beta = 0.0

        if data_mean is not None and data_std is not None:
            self.lowest = float(np.min((0 - data_mean) / data_std))
            self.highest = float(np.max((1 - data_mean) / data_std))

        self.cov_calculator = CovarianceCalculator()
        self.register_buffer('pattern', torch.zeros(self.conv.weight.shape))

        # initialize parameters
        nn.init.xavier_uniform_(self.conv.weight.data)
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        self.X = x

        out = self.conv.forward(x)
        self.pre_activation = out

        if self.activation_fn:
            out = self.activation_fn(out)

        return out

    def learn_pattern(self, x):
        y = self.forward(x)
        x = torch.nn.functional.unfold(x, kernel_size=self.conv.kernel_size, dilation=self.conv.dilation,
                                       padding=self.conv.padding, stride=self.conv.stride)

        bs, n_patches = x.shape[0], x.shape[2]
        x = x.permute(0, 2, 1).reshape(bs*n_patches, -1)
        y_in = self.pre_activation.permute(0, 2, 3, 1).reshape(bs * n_patches, -1)

        if self.activation_fn:
            cond = y_in > 0
        else:
            cond = torch.ones_like(y_in)

        self.cov_calculator.add_batch(x, y_in, cond)

        return y

    def compute_pattern(self, debug=False):
        cov_xy = self.cov_calculator.compute()
        var_y = torch.sum(self.conv.weight.reshape(self.conv.weight.size(0), -1).t() * cov_xy, dim=0).unsqueeze(0)

        self.pattern = (cov_xy / (var_y + (var_y == 0).float())).transpose(0, 1).reshape(self.conv.weight.shape)

        if debug:
            self.cov_xy = cov_xy
            self.var_y = var_y

    def analyze(self, method, R):
        # if previous layer was a dense layer, R needs to be reshaped
        # to the form of self.X after the convolution in the forward pass
        batch_size, _, height, width = self.pre_activation.shape
        if R.shape != torch.Size([batch_size, self.conv.out_channels, height, width]):
            R = R.view(batch_size, self.conv.out_channels, height, width)

        if method == ExplainingMethod.lrp:
            return self._lrp_backward(R)
        elif method == ExplainingMethod.gradient or method == ExplainingMethod.grad_times_input:
            return self._gradient_backward(R)
        elif method == ExplainingMethod.guided_backprop:
            return self._guided_backprop_backward(R)
        elif method == ExplainingMethod.pattern_attribution:
            return self._pattern_attribution_backward(R)

    def _guided_backprop_backward(self, R):
        if self.activation_fn is not None:
            if hasattr(self.activation_fn, "beta"):
                R = torch.nn.functional.softplus(R, self.activation_fn.beta) * torch.sigmoid(self.activation_fn.beta * self.pre_activation)
            else:
                R = torch.nn.functional.relu(R) * (self.pre_activation >= 0).float()

        newR = self.deconvolve(R, self.conv.weight)
        return newR

    def _gradient_backward(self, R):
        if self.activation_fn is not None:
            if hasattr(self.activation_fn, "beta"):
                R = R * torch.sigmoid(self.activation_fn.beta * self.pre_activation)
            else:
                R = R * (self.pre_activation >= 0).float()

        newR = self.deconvolve(R, self.conv.weight)

        return newR

    def _pattern_attribution_backward(self, R):
        if self.activation_fn is not None:
            if hasattr(self.activation_fn, "beta"):
                R = R * torch.sigmoid(self.activation_fn.beta * self.pre_activation)
            else:
                R = R * (self.pre_activation >= 0).float()

        if self.pattern is None:
            raise RuntimeError('Pattern need to be set in order to use pattern attribution.')

        newR = self.deconvolve(R, self.conv.weight*self.pattern)

        return newR

    def set_lrp_rule(self, lrp_rule, alpha=None, beta=None):
        if alpha is not None and beta is not None:
            self.alpha = alpha
            self.beta = beta
        self.lrp_rule = lrp_rule

    def _lrp_backward(self, R):

        if self.lrp_rule == LRPRule.z_b:
            return self._lrp_zb(R)
        elif self.lrp_rule == LRPRule.alpha_beta:
            return self._lrp_alpha_beta(R)
        elif self.lrp_rule == LRPRule.z_plus:
            return self._lrp_zp(R)

    def _lrp_alpha_beta(self, R):
        _, _, height_filter, width_filter = self.conv.weight.shape

        p_weights = self.conv.weight.clamp(min=0)
        n_weights = self.conv.weight.clamp(max=0)

        ZA = torch.nn.functional.conv2d(input=self.X, weight=p_weights, bias=None, padding=self.conv.padding,
                                        stride=self.conv.stride, groups=self.conv.groups,
                                        dilation=self.conv.dilation) + 1e-9
        ZB = torch.nn.functional.conv2d(input=self.X, weight=n_weights, bias=None, padding=self.conv.padding,
                                        stride=self.conv.stride, groups=self.conv.groups,
                                        dilation=self.conv.dilation) + 1e-9

        SA = self.alpha * R / ZA
        SB = -self.beta * R / ZB

        newR = self.X * (self.deconvolve(SA, p_weights) + self.deconvolve(SB, n_weights))

        return newR

    def _lrp_zb(self, R):

        weights = self.conv.weight
        p_weights = self.conv.weight.clamp(min=0)
        n_weights = self.conv.weight.clamp(max=0)

        L, H = self.lowest * torch.ones_like(self.X), self.highest * torch.ones_like(self.X)

        Z = torch.nn.functional.conv2d(input=self.X, weight=weights, bias=None, padding=self.conv.padding,
                                       stride=self.conv.stride, groups=self.conv.groups, dilation=self.conv.dilation) \
            - torch.nn.functional.conv2d(input=L, weight=p_weights, bias=None, padding=self.conv.padding,
                                         stride=self.conv.stride, groups=self.conv.groups, dilation=self.conv.dilation) \
            - torch.nn.functional.conv2d(input=H, weight=n_weights, bias=None, padding=self.conv.padding,
                                         stride=self.conv.stride, groups=self.conv.groups, dilation=self.conv.dilation) \
            + 1e-9

        S = R / Z

        newR = self.X * self.deconvolve(S, weights) - L * self.deconvolve(S, p_weights) - H * self.deconvolve(S,
                                                                                                              n_weights)

        return newR

    def _lrp_zp(self, R):

        weights = self.conv.weight.clamp(min=0)

        Z = torch.nn.functional.conv2d(input=self.X, weight=weights, bias=None, padding=self.conv.padding,
                                       stride=self.conv.stride, groups=self.conv.groups,
                                       dilation=self.conv.dilation) + 1e-9

        S = R / Z
        newR = self.X * self.deconvolve(S, weights)

        return newR

    def deconvolve(self, y, weights):

        # dimensions before convolution in forward pass
        # the deconvolved image has to have the same dimension
        _, _, org_height, org_width = self.X.shape

        # stride and padding from forward convolution
        padding = self.conv.padding
        stride = self.conv.stride

        _, _, filter_height, filter_width = weights.shape

        # the deconvolved image has minimal size
        # to obtain an image with the same size as the image before the convolution in the forward pass
        # we pad the output of the deconvolution
        output_padding = ((org_height + 2 * padding[0] - filter_height) % stride[0],
                          (org_width + 2 * padding[1] - filter_width) % stride[1])  # a=(i+2p−k) mod s

        # perform actual deconvolution
        # this is basically a forward convolution with flipped (and permuted) filters/weights
        deconvolved = torch.nn.functional.conv_transpose2d(input=y, weight=weights, bias=None,
                                                           padding=self.conv.padding, stride=self.conv.stride,
                                                           groups=self.conv.groups, dilation=self.conv.dilation,
                                                           output_padding=output_padding)

        return deconvolved
