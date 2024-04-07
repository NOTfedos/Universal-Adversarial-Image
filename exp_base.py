
import numpy as np
from matplotlib import pyplot as plt
import math

from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.transforms import v2

from torch.utils.tensorboard import SummaryWriter


norm = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class FocalLoss(torch.nn.Module):
    '''
    Multi-class Focal loss implementation
    '''

    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inp, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = torch.nn.functional.log_softmax(inp, dim=1)
        pt = torch.exp(logpt)
        logpt = ((1 - pt) ** self.gamma) * logpt
        loss = torch.nn.functional.nll_loss(logpt, target, self.weight)
        return loss


def get_accuracy(model, univ_image, num_classes, mask):
    """
        Validating method
        returns count of correct predictions and their probabilities
    """

    with torch.no_grad():
        probs = []
        k = 0
        for label in range(num_classes):
            masked = mask(label, num_classes)(univ_image).unsqueeze(0)
            preds = model(norm(masked))
            prob, pred_label = torch.nn.Softmax(dim=1)(preds).max(1)
            if pred_label[0] == label:
                k += 1
            probs.append(prob[0].cpu().detach().numpy())

        return k, probs


def train(model, univ_image, epochs, batch_size, num_classes, criterion, mask, writer=None) -> tuple[torch.Tensor, int]:
    """
        Train method. Returns configured univ_image
    """
    opt = optim.SGD([univ_image], lr=1e-1)
    # criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=250, min_lr=1e-4, threshold=1e-4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    cur_class = 0
    start_time = datetime.now()
    for t in range(epochs):

        batch = []
        labels = []
        for k in range(batch_size):
            batch.append(mask(cur_class, num_classes)(univ_image).to(device))
            labels.append(cur_class)
            cur_class = (cur_class + 1) % num_classes
        batch = torch.stack(batch, dim=0).to(device)
        labels = torch.LongTensor(labels).to(device)

        preds = model(norm(batch))
        loss = criterion(preds, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        scheduler.step(loss.item())

        univ_image.data.clamp_(0, 1)

        acc, probs = get_accuracy(model, univ_image, num_classes, mask)
        if t % (epochs // 10) == 0:
            # print(preds.shape, labels.shape)

            delta = datetime.now() - start_time
            start_time = datetime.now()

            print(
                f"EPOCH = {t}/{epochs} | loss = {loss.item()} | acc={acc / num_classes} | lr={scheduler.optimizer.param_groups[0]['lr']} | time={int(delta.total_seconds() // 60)}:{delta.seconds % 60}")
            if writer is not None:
                writer.add_scalar(f'Acc', acc / num_classes, t)

        if writer is not None:
            writer.add_scalar(f'Loss', loss.item(), t)
            writer.add_scalar(f'LR', scheduler.optimizer.param_groups[0]['lr'], t)
            writer.flush()

        if acc == num_classes:
            print(
                f"CLOSING EPOCH = {t}/{epochs} | loss = {loss.item()} | acc={acc / num_classes} | lr={scheduler.optimizer.param_groups[0]['lr']} | time={int(delta.total_seconds() // 60)}:{delta.seconds % 60}")
            return univ_image, t

    return univ_image, epochs


def plot_probs(probs, NUM_CLASSES):
    """
        Plots the probability distribution of pridicted labels
    """
    plt.plot(np.arange(NUM_CLASSES), probs, label="origin prediction probs", color="blue")
    plt.plot(np.arange(NUM_CLASSES), [sum(probs) / NUM_CLASSES] * NUM_CLASSES,
             label="mean={:.2f}".format(sum(probs) / NUM_CLASSES), color="red")
    plt.xlabel('Class')
    plt.ylabel("Probability")
    plt.title("Prediction prob for each class")
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.legend()
    plt.show()


def init_image(image_size):
    """
        Init randomly image
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.randn(size=(3, image_size, image_size), requires_grad=True, device=device)
