import torch
import torch.nn.functional as F

from tqdm import tqdm

class WNet(torch.nn.Module):

    def __init__(self, feature_dim, num_groups):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_groups = num_groups

        self.model = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, num_groups, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return x


def calibrate_with_tau_and_w_logits(logits,
                                    features,
                                    tau,
                                    hard,
                                    w_net=None,
                                    w_logits=None):
    assert (w_logits is not None) != (w_net is not None)
    
 
    N, num_classes = logits.shape
    if hard:
        num_groups = w_net.num_groups
        group_log_softmax = torch.log_softmax(
            w_net(features), dim=1)
        group_argmax = torch.argmax(group_log_softmax, dim=1)
        group_hard_prob = F.one_hot(
            group_argmax, num_classes=num_groups).view((N, num_groups, 1))
        group_hard_prob = group_hard_prob.expand((N, num_groups, num_classes))

        temp_logits = logits.view((N, 1, num_classes)) / \
            tau.view((1, num_groups, 1))
        temp_log_softmax = torch.log_softmax(temp_logits, dim=2)
        calibrated_logits = torch.sum(
            temp_log_softmax * group_hard_prob, dim=1)
        return calibrated_logits
    else:

        if w_logits is not None:
            num_groups = w_logits.shape[1]
            group_log_softmax = torch.log_softmax(
                w_logits, dim=1).view((N, num_groups, 1))
        else:
            num_groups = w_net.num_groups
            group_log_softmax = torch.log_softmax(
                w_net(features), dim=1).view((N, num_groups, 1))

        group_log_softmax = group_log_softmax.expand(
            (N, num_groups, num_classes))
        temp_logits = logits.view((N, 1, num_classes)) / \
            tau.view((1, num_groups, 1))
        temp_log_softmax = torch.log_softmax(temp_logits, dim=2)
        calibrated_logits = torch.logsumexp(group_log_softmax +
                                            temp_log_softmax, dim=1)
        return calibrated_logits


def optimize_group_fn(
        features,
        logits,
        labels,
        w_net,
        hard_group,
        method_config):

    if isinstance(w_net, str):
        train_w = True
        assert isinstance(method_config.num_groups, int)
        w_net = WNet(feature_dim=features.shape[1],
                     num_groups=method_config.num_groups)
    else:
        train_w = False
        assert isinstance(w_net, torch.nn.Module)

    tau = torch.nn.Parameter(torch.tensor(
        [1.5] * method_config.num_groups, 
        requires_grad=True, device=features.device))

    if train_w:
        params = [tau] + list(w_net.parameters())
    else:
        params = [tau]

    if method_config.optimizer.name == "lbfgs" or not train_w:
        optimizer = torch.optim.LBFGS(params,
                                      line_search_fn="strong_wolfe",
                                      max_iter=method_config.optimizer.steps)
    else:
        raise ValueError(method_config.optimizer)

    W_gpu = w_net.to(features.device)

    def closure():
        optimizer.zero_grad()
        
        # Calculate weight decay loss
        reg_weight_decay = 0
        for name, param in W_gpu.named_parameters():
            if "weight" in name:
                reg_weight_decay += torch.mean((param)**2)
        reg_weight_decay_loss = reg_weight_decay * method_config.w_net.weight_decay

        # Calculate NLL loss
        calibrated_logits = calibrate_with_tau_and_w_logits(
            logits=logits,
            features=features,
            tau=tau,
            w_net=W_gpu,
            hard=hard_group
        )

        main_loss = F.cross_entropy(calibrated_logits, labels)

        # Gather all loss
        _loss = main_loss + reg_weight_decay_loss

        _loss.backward()
        return _loss

    optimizer.step(closure=closure)
            
    return tau.detach().cpu(), w_net.cpu()


def train_partitions(features,
                     logits,
                     labels,
                     w_net,
                     method_config):
    w_net_list = []
    print("Generating partitions...")
    for partition_i in tqdm(range(method_config.num_partitions)):
        trained_tau, trained_w_net = optimize_group_fn(features.to("cuda:0"),
                                                       logits.to("cuda:0"),
                                                       labels.to("cuda:0"),
                                                       hard_group=False,
                                                       w_net=w_net,
                                                       method_config=method_config)
        w_net_list.append(trained_w_net)
    return w_net_list


def calibrate_combine(val_features,
                      val_logits,
                      val_labels,
                      test_train_features,
                      test_train_logits,
                      test_train_labels,
                      test_test_features,
                      test_test_logits,
                      method_config,
                      base_calibrate_fn,
                      seed,
                      cfg,
                      *args, **kwargs):
    
    w_net_list = train_partitions(val_features,
                                val_logits,
                                val_labels,
                                w_net=method_config.w_net.model,
                                method_config=method_config)
    
    calibrated_probs = []
    print("Calibrating with partitions...")
    for trained_w_net in tqdm(w_net_list):

        train_group_logits = trained_w_net(test_train_features)
        test_group_logits = trained_w_net(test_test_features)
        # Hard group
        train_groups_id = torch.argmax(
            train_group_logits, dim=1)
        test_groups_id = torch.argmax(
            test_group_logits, dim=1)

        _calibrated_probs = torch.zeros_like(test_test_logits)
        for _g in range(method_config.num_groups):
            train_group_mask = train_groups_id == _g
            test_group_mask = test_groups_id == _g
            group_train_logits = test_train_logits[train_group_mask]
            group_train_labels = test_train_labels[train_group_mask]

            group_test_logits = test_test_logits[test_group_mask]

            _group_calibrated_results = base_calibrate_fn(
                method_name=method_config.base_calibrator.name,
                train_logits=group_train_logits,
                train_labels=group_train_labels,
                test_logits=group_test_logits
            )
            if "prob" in _group_calibrated_results:
                _group_calibrated_prob = _group_calibrated_results["prob"]
            else:
                _group_calibrated_prob = torch.softmax(_group_calibrated_results["logits"],
                                                        dim=1)
            _calibrated_probs[test_group_mask] = _group_calibrated_prob

        calibrated_probs.append(_calibrated_probs.detach())
    calibrated_probs = torch.stack(calibrated_probs, dim=0).mean(0)

    return {
        "prob": calibrated_probs
    }
