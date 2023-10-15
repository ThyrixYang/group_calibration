import logging
import torch
import torch.nn.functional as F
import numpy as np


def calibrate(train_logits,
              train_labels,
              test_logits,
              *args, **kwargs):
    train_logits = train_logits.cuda()
    train_labels = train_labels.cuda()

    tau = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.LBFGS([tau],
                                  line_search_fn="strong_wolfe",
                                  max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(train_logits / tau, train_labels)
        loss.backward()
        return loss
    optimizer.step(closure=closure)
    final_loss = closure()

    if torch.isnan(tau):
        tau = 1
    else:
        tau = tau.item()

    return {
        "tau": tau,
        "logits": test_logits / tau,
        "loss": final_loss.item()
    }
