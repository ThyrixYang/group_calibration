import torch
import pickle
import logging

from utils import RandomSplitter

def load_data(data_config,
              test_splits=(0.1, 0.9),
              seed=None):
    with open(data_config.val_path, "rb") as f:
        val_data = pickle.load(f)

    with open(data_config.test_path, "rb") as f:
        test_data = pickle.load(f)

    val_acc = (torch.argmax(val_data["logits"], dim=1)
               == val_data["labels"]).float().mean().item()
    test_acc = (torch.argmax(test_data["logits"], dim=1)
                == test_data["labels"]).float().mean().item()
    logging.info("Dataset: val_acc: {:.4f}, test_acc: {:.4f}".format(val_acc, test_acc))

    test_splitter = RandomSplitter(splits=test_splits,
                                   num=test_data["logits"].shape[0],
                                   seed=seed)
    test_train_data, test_test_data = {}, {}
    test_train_data["logits"], test_test_data["logits"] = test_splitter.split(
        test_data["logits"])
    test_train_data["labels"], test_test_data["labels"] = test_splitter.split(
        test_data["labels"]
    )
    test_train_data["features"], test_test_data["features"] = test_splitter.split(
        test_data["features"]
    )
    return val_data, test_train_data, test_test_data