import enum


class LRPRule(enum.Enum):
    z_plus = 0
    z_b = 1
    alpha_beta = 2
    alpha_beta_bias = 3


class ExplainingMethod(enum.Enum):
    lrp = 0
    gradient = 1
    guided_backprop = 2
    integrated_grad = 3
    pattern_attribution = 4
    grad_times_input = 5
