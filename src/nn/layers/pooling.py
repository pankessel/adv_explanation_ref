import torch.nn as nn
import torch


class MaxPool(nn.Module):
    def __init__(self, kernel_size=2, stride=None, padding=0, dilation=1, ceil_mode=False):
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                 ceil_mode=ceil_mode, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size, stride, padding)
        self.indices = None

    def forward(self, x):
        self.X = x

        output, self.indices = self.pool(x)

        return output

    def learn_pattern(self, x):
        return self.forward(x)

    def compute_pattern(self):
        pass

    def gradient_backward(self, R):
        return self.lrp_backward(R)

    def lrp_backward(self, R):

        batch_size, channels, height, width = self.X.shape
        height = int(height/2)
        width = int(width/2)

        if R.shape != torch.Size([batch_size, channels, height, width]):
            R = R.view(batch_size, channels, height, width)

        return self.unpool(R, self.indices)

    def analyze(self, method, R):

        batch_size, channels, height, width = self.X.shape
        height = int(height/2)
        width = int(width/2)

        if R.shape != torch.Size([batch_size, channels, height, width]):
            R = R.view(batch_size, channels, height, width)

        return self.unpool(R, self.indices)


class SumPool(nn.Module):
    def __init__(self, kernel_size=2, stride=None, padding=0, dilation=1, ceil_mode=False):
        super(SumPool, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

    def forward(self, x):
        self.X = x

        output = self.pool(x)

        return output

    def learn_pattern(self, x):
        return self.forward(x)

    def compute_pattern(self):
        pass

    def lrp_backward(self, R):

        Z = (self.forward(self.X)+1e-9)
        S = R / Z
        C = torch.zeros(self.X.shape)
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            C[:, i::2, j::2, :] += S * 0.25
        R = self.X*C
        return R
