import torch
import torch.nn as nn
import math


class MaskLine(nn.Module):
    def __init__(self, class_ind, count_classes, axis=0):
        super(MaskLine, self).__init__()
        self.class_ind = class_ind
        self.count_classes = count_classes
        self.axis = axis

    def forward(self, image):
        c, h, w = image.shape

        mask = torch.ones_like(image)

        if self.axis == 0:
            interval = h // self.count_classes
            mask[:, self.class_ind * interval: (self.class_ind + 1) * interval, :] = torch.zeros([3, interval, w])
        elif self.axis == 1:
            interval = w // self.count_classes
            mask[:, :, self.class_ind * interval: (self.class_ind + 1) * interval] = torch.zeros([3, h, interval])

        return image * mask
