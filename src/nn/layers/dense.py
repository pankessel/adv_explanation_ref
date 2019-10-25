from ..enums import ExplainingMethod, LRPRule
from ..utils import CovarianceCalculator

import torch
import torch.nn as nn
import numpy as np


class Dense(nn.Module):
    def __init__(self, in_dim, out_dim, activation_fn=None, lrp_rule=LRPRule.alpha_beta,
                 data_mean=0.0,
                 data_std=1.0, BN=False):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.batch_norm = None
        if BN:
            self.batch_norm = nn.BatchNorm1d(out_dim)
        self.activation_fn = activation_fn

        self.X = 0
        self.pre_activation = 0
        self.lrp_rule = lrp_rule
        self.alpha = 1.0
        self.beta = 0.0

        self.lowest = np.min((0 - data_mean) / data_std)
        self.highest = np.max((1 - data_mean) / data_std)

        self.cov_calculator = CovarianceCalculator()
        self.register_buffer('pattern', torch.zeros(out_dim, in_dim))

        # initialize parameters
        nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        self.X = x
        out = self.linear(x)

        if self.batch_norm is not None:
            out = self.batch_norm(out)

        self.pre_activation = out

        if self.activation_fn:
            out = self.activation_fn(out)

        return out

    def learn_pattern(self, x):
        y = self.forward(x)
        y_in = self.pre_activation
        if self.activation_fn:
            cond = y_in > 0
        else:
            cond = torch.ones_like(y_in)

        self.cov_calculator.add_batch(self.X, y_in, cond)

        return y

    def compute_pattern(self, debug=False):
        cov_xy = self.cov_calculator.compute()
        var_y = torch.sum(self.linear.weight.t()*cov_xy, dim=0).unsqueeze(0)

        self.pattern = (cov_xy / (var_y + (var_y == 0).float())).transpose(1, 0)

        if debug:
            self.cov_xy = cov_xy
            self.var_y = var_y

    def analyze(self, method, R):
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
                R = torch.nn.functional.softplus(R, beta=self.activation_fn.beta) * torch.sigmoid(self.activation_fn.beta * self.pre_activation)
            else:
                R = torch.nn.functional.relu(R) * (self.pre_activation >= 0).float()

        weight = self.linear.weight
        if self.batch_norm is not None:
            weight = weight * self.batch_norm.weight.unsqueeze(1) / torch.sqrt(
                self.batch_norm.running_var.unsqueeze(1) + self.batch_norm.eps)

        newR = torch.matmul(R, weight)

        return newR

    def _gradient_backward(self, R):
        return self._generic_backward(R, self.linear.weight)

    def _pattern_attribution_backward(self, R):
        if self.pattern is None:
            raise RuntimeError('Pattern needs to be set in order to use pattern attribution.')

        return self._generic_backward(R, self.linear.weight * self.pattern)

    def _generic_backward(self, R, weight):
        if self.activation_fn is not None:
            if hasattr(self.activation_fn, "beta"):
                R = R * torch.sigmoid(self.activation_fn.beta * self.pre_activation)
            else:
                R = R * (self.pre_activation >= 0).float()
        if self.batch_norm is not None:
            weight = weight * self.batch_norm.weight.unsqueeze(1) / torch.sqrt(
                self.batch_norm.running_var.unsqueeze(1) + self.batch_norm.eps)

        newR = torch.matmul(R, weight)

        return newR

    def set_lrp_rule(self, lrp_rule, alpha=None, beta=None):
        if alpha is not None and beta is not None:
            self.alpha = alpha
            self.beta = beta
        self.lrp_rule = lrp_rule

    def _lrp_backward(self, R):
        if self.lrp_rule == LRPRule.z_b:
            return self._lrp_zb(R)
        elif self.lrp_rule == LRPRule.z_plus:
            return self._lrp_zp(R)
        else:
            return self._lrp_alpha_beta(R)

    def _lrp_zp(self, R):

        weight = self.linear.weight

        if self.batch_norm is not None:
            weight = weight * self.batch_norm.weight.unsqueeze(1) / torch.sqrt(
                self.batch_norm.running_var.unsqueeze(1) + self.batch_norm.eps)

        V = weight.clamp(min=0.0)

        Z = self.X.mm(V.t()) + 1e-9
        S = R / Z
        C = S.mm(V)
        R = self.X * C
        return R

    def _lrp_alpha_beta(self, R):

        X = self.X

        weight = self.linear.weight

        if self.batch_norm is not None:
            weight = weight * self.batch_norm.weight.unsqueeze(1) / torch.sqrt(
                self.batch_norm.running_var.unsqueeze(1) + self.batch_norm.eps)

        V = weight.clamp(min=0.0)
        Z = X.mm(V.t()) + 1e-9
        S = R / Z
        C = S.mm(V)
        RP = self.alpha * X * C

        V = weight.clamp(max=0.0)
        Z = X.mm(V.t()) + 1e-9
        S = R / Z
        C = S.mm(V)
        RM = self.beta * X * C
        Rnew = RP - RM

        return Rnew

    def _lrp_zb(self, R):

        weight = self.linear.weight

        if self.batch_norm is not None:
            weight = weight * self.batch_norm.weight.unsqueeze(1) / torch.sqrt(
                self.batch_norm.running_var.unsqueeze(1) + self.batch_norm.eps)

        W = weight.t()
        V, U = W.clamp(min=0), W.clamp(max=0)
        X, L, H = self.X + 0, torch.ones_like(self.X.data) * self.lowest, torch.ones_like(self.X.data) * self.highest

        Z = torch.mm(X, W) - torch.mm(L, V) - torch.mm(H, U) + 1e-9
        S = R / Z
        Rnew = X * torch.mm(S, W.t()) - L * torch.mm(S, V.t()) - H * torch.mm(S, U.t())
        return Rnew
