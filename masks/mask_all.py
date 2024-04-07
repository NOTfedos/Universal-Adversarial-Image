import torch
import torch.nn as nn
import math


class MaskImageAll(nn.Module):
    def __init__(self, class_ind, count_classes):
        super(MaskImageAll, self).__init__()
        self.class_ind = class_ind
        self.count_classes = count_classes

    def forward(self, image):
        c, h, w = image.shape

        num_columns = int(math.ceil(math.sqrt(self.count_classes)))
        num_rows = int(math.ceil(self.count_classes / float(num_columns)))

        tile_w, tile_h = int(math.floor(w / num_columns)), int(math.floor(h / num_rows))

        row = self.class_ind // num_columns
        column = self.class_ind % num_columns

        mask = torch.ones_like(image)  # maybe add requires_grad = Ture
        mask[:, row * tile_h: (row + 1) * tile_h, column * tile_w: (column + 1) * tile_w] = torch.zeros([3, tile_h, tile_w])

        return image * mask
