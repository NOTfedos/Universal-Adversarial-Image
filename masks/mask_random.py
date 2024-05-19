import torch
import torch.nn as nn
import math
from random import randint


class MaskRandomPixels(nn.Module):
    def __init__(self, count_classes, img_shape, mask_ratio=0.15, device=torch.device('cuda')):
        super(MaskRandomPixels, self).__init__()
        self.count_classes = count_classes
        self.mask_ratio = mask_ratio

        self.masks = []
        i = 0
        while i < count_classes:
            mask = torch.ones((3, img_shape, img_shape))
            mask_inds = []
            j = 0
            while j < int(mask_ratio * img_shape * img_shape):
                random_x, random_y = randint(0, img_shape - 1), randint(0, img_shape - 1)
                if (random_x, random_y) in mask_inds:
                    continue
                mask[:, random_x, random_y] = torch.zeros((3,))
                mask_inds.append((random_x, random_y))
                j += 1

            # Check if mask already applied to another class_ind
            mask = mask.to(device)
            to_continue = False
            for mask_ in self.masks:
                if torch.allclose(mask_, mask):
                    to_continue=True
                    break
            if to_continue:
                continue

            mask.requires_grad_(True)
            self.masks.append(mask)
            i += 1

    def forward(self, image, class_ind):
        return image * self.masks[class_ind]


if __name__ == '__main__':
    MaskRand = MaskRandomPixels(5, 3)
    for i in range(5):
        print("-----------------------------{i}------------------".format(i=i))
        image = torch.randn((3, 3, 3)).to(torch.device('cuda'))
        print("image:")
        print(image, "\n")
        print("masked:\n")
        print(MaskRand(image, i))
