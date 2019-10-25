import torch
from .enums import ExplainingMethod
import numpy as np
import os
import math
import torchvision
from PIL import Image

import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt


def plot_overview(images, heatmaps, mean, std,
                  captions=['Target Image', 'Original Image', 'Manipulated Image', 'Target Explanation', 'Original Explanation', 'Manipulated Explanation'],
                  filename="overview.png", images_per_row=3):
    """
    Helper method for plotting the result of the attack
    """
    plots = [torch_to_image(img, mean, std) for img in images] + [heatmap_to_image(heatmap) for heatmap in heatmaps]

    img_cmap = 'jet'
    heatmap_cmap = 'jet' if len(plots[-1].shape) == 3 else 'coolwarm'
    cmaps = [img_cmap] * len(images) + [heatmap_cmap] * len(heatmaps)

    plot_grid(plots, captions, cmap=cmaps, filename=filename, images_per_row=images_per_row)


def load_image(data_mean, data_std, device, image_name):
    """
    Helper method to load an image into a torch tensor. Includes preprocessing.
    """
    im = Image.open(image_name)
    x = torchvision.transforms.Normalize(mean=data_mean, std=data_std)(
        torchvision.transforms.ToTensor()(
            torchvision.transforms.CenterCrop(224)(torchvision.transforms.Resize(256)(im))))
    x = x.unsqueeze(0).to(device)
    return x


def clamp(x, mean, std):
    """
    Helper method for clamping the adversarial example in order to ensure that it is a valid image
    """
    upper = torch.from_numpy(np.array((1.0 - mean) / std)).to(x.device)
    lower = torch.from_numpy(np.array((0.0 - mean) / std)).to(x.device)

    if x.shape[1] == 3:  # 3-channel image
        for i in [0, 1, 2]:
            x[0][i] = torch.clamp(x[0][i], min=lower[i], max=upper[i])
    else:
        x = torch.clamp(x, min=lower[0], max=upper[0])
    return x


def get_expl(model, x, method, desired_index=None):
    """
    Helper method to get the heatmap
    """
    x.requires_grad = True
    acc, class_idx = model.classify(x)
    if desired_index is None:
        desired_index = class_idx

    if method == ExplainingMethod.integrated_grad:
        # calculate the integrand in one batch
        # we use DataParallel mode of model to fit the batch in memory of (multiple) gpu(s)
        num_summands = 30
        prefactors = x.new_tensor([k / num_summands for k in range(1, num_summands + 1)])
        parallel_model = torch.nn.DataParallel(model)
        y = parallel_model(prefactors.view(num_summands, 1, 1, 1) * x)

        # we sum the result and then take the derivative (instead of summing derivatives as in most implementations),
        # (d/dx) (n*y_1(1/n*x) + n/2*y_1(2/n*x) .... + y_n(x) ) = y_1'+....y'_n
        y = torch.nn.functional.softmax(y, 1)[:, int(desired_index)]
        y = (1 / num_summands) * torch.sum(y / prefactors, dim=0)
        heatmap = torch.autograd.grad(y, x, create_graph=True)[0]
    else:
        heatmap = model.analyze(method=method, R=None, index=desired_index)

    if method == ExplainingMethod.grad_times_input or method == ExplainingMethod.integrated_grad:
        heatmap = heatmap * x

    heatmap = torch.sum(torch.abs(heatmap), dim=1)

    normalized_heatmap = heatmap / torch.sum(heatmap)

    return normalized_heatmap, acc, class_idx


def torch_to_image(tensor, mean=0, std=1):
    """
    Helper function to convert torch tensor containing input data into image.
    """
    if len(tensor.shape) == 4:
        img = tensor.permute(0, 2, 3, 1)

    img = img.contiguous().squeeze().detach().cpu().numpy()

    img = img * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
    return np.clip(img, 0, 1)


def heatmap_to_image(heatmap):
    """
    Helper image to convert torch tensor containing a heatmap into image.
    """
    if len(heatmap.shape) == 4:
        heatmap = heatmap.permute(0, 2, 3, 1)

    img = heatmap.squeeze().data.cpu().numpy()

    img = img / np.max(np.abs(img))  # divide by maximum
    img = np.maximum(-1, img)
    img = np.minimum(1, img) * 0.5  # clamp to -1 and divide by two -> range [-0.5, 0.5]
    img = img + 0.5

    return img


def make_dir(directory_name):
    if not directory_name:
        return ''
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    if directory_name[-1] != '/':
        directory_name = directory_name + '/'

    return directory_name


def plot_grid(images, titles=None, images_per_row=3, cmap='gray', norm=mpl.colors.NoNorm(), filename="overview.png"):
    """
    Helper method to plot a grid with matplotlib
    """
    plt.close("all")
    num_images = len(images)
    images_per_row = min(num_images, images_per_row)

    num_rows = math.ceil(num_images / images_per_row)

    if len(cmap) != num_images or type(cmap) == str:
        cmap = [cmap] * num_images

    fig, axes = plt.subplots(nrows=num_rows, ncols=images_per_row)

    fig = plt.gcf()
    fig.set_size_inches(4 * images_per_row, 5 * int(np.ceil(len(images) / images_per_row)))
    for i in range(num_rows):
        for j in range(images_per_row):

            idx = images_per_row * i + j

            if num_rows == 1:
                a_ij = axes[j]
            elif images_per_row == 1:
                a_ij = axes[i]
            else:
                a_ij = axes[i, j]
            a_ij.axis('off')
            if idx >= num_images:
                break
            a_ij.imshow(images[idx], cmap=cmap[idx], norm=norm, interpolation='nearest')
            a_ij.set_title(titles[idx])

    plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)

    plt.savefig(filename)
    plt.close()


class CovarianceCalculator:
    """
    Helper class to calculate covariance for pattern attribution.
    """

    def __init__(self):
        self.mean_x = 0
        self.mean_y = 0
        self.mean_xy = 0
        self.n = 0
        self.n_cond = 0

    def add_batch(self, x, y, cond):
        cond = cond.float()

        batch_size = x.shape[0]
        batch_size_cond = torch.sum(cond, dim=0)

        self.n += batch_size
        self.n_cond += batch_size_cond

        self.mean_y = (self.mean_y * (self.n - batch_size) + torch.sum(y, dim=0)) / self.n
        self.mean_xy = self._add_to_cond_running_average(self.mean_xy,
                                                         torch.sum(torch.bmm(x.unsqueeze(2), (cond * y).unsqueeze(1)),
                                                                   dim=0), batch_size_cond)
        self.mean_x = self._add_to_cond_running_average(self.mean_x,
                                                        torch.sum(torch.bmm(x.unsqueeze(2), cond.unsqueeze(1)), dim=0),
                                                        batch_size_cond)

    def compute(self):
        return self.mean_xy - self.mean_y * self.mean_x

    def _add_to_cond_running_average(self, prev, batch, bs_cond):
        # ensures that we do not divide by zero
        n_cond_reg = self.n_cond + (self.n_cond == 0).float()

        return (prev * (n_cond_reg.unsqueeze(0) - bs_cond.unsqueeze(0)) + batch) / (n_cond_reg.unsqueeze(0))
