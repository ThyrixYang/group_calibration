import torch
import numpy as np
import torch.nn.functional as F
import torchmetrics.functional as tmF

def evaluate(y, num_classes, n_bins=15, pred_prob=None, pred_logits=None):
    if pred_logits is not None:
        pred_logits = pred_logits.contiguous()
    if pred_prob is None:
        pred_prob = torch.softmax(pred_logits, dim=1)
    ece_multiclass = tmF.calibration_error(pred_prob,
                                           y,
                                           task="multiclass",
                                           n_bins=n_bins,
                                           num_classes=num_classes)
    if pred_logits is not None:
        nll = F.cross_entropy(pred_logits, y).item()
    else:
        nll = -torch.mean(torch.sum(torch.log(pred_prob + 1e-10)
                                    * F.one_hot(y, num_classes=num_classes), dim=1)).item()
    pred_labels = torch.argmax(pred_prob, dim=1)
    acc = (pred_labels == y).float().mean().item()
    results = {
        "ece_m": ece_multiclass.item(),
        "nll": nll,
        "acc": acc,
    }
    return results