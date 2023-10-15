import functools

import torch
import methods.temp_scaling
import methods.group_calibration
import methods.nn_calibration
import methods.mix_calibration


def get_calibrate_fn(method_config):
    if method_config.name in ["temp_scaling"]:
        return methods.temp_scaling.calibrate
    elif method_config.name in ["histogram_binning",
                                "isotonic_regression"]:
        return methods.nn_calibration.calibrate
    elif method_config.name in ["ets"]:
        return methods.mix_calibration.calibrate
    else:
        raise ValueError("config_name {} not found".format(method_config.name))


def calibrate(method_config,
              val_data,
              test_train_data,
              test_test_data,
              seed,
              cfg):
    if method_config.name == "none":
        return {
            "logits": test_test_data["logits"]
        }

    train_set = method_config.get("train_set", "test_train")
    if train_set == "val":
        train_logits = val_data["logits"]
        train_labels = val_data["labels"]
    elif train_set == "test_train":
        train_logits = test_train_data["logits"]
        train_labels = test_train_data["labels"]
    else:
        assert train_set == "val+test_train"
        train_logits = torch.cat(
            [val_data["logits"], test_train_data["logits"]], dim=0)
        train_labels = torch.cat(
            [val_data["labels"], test_train_data["labels"]], dim=0)

    test_test_logits = test_test_data["logits"]

    if "group_calibration_combine" in method_config.name:
        return methods.group_calibration.calibrate_combine(val_features=val_data["features"],
                                                           val_logits=val_data["logits"],
                                                           val_labels=val_data["labels"],
                                                           test_train_features=test_train_data["features"],
                                                           test_train_logits=test_train_data["logits"],
                                                           test_train_labels=test_train_data["labels"],
                                                           test_test_features=test_test_data["features"],
                                                           test_test_logits=test_test_data["logits"],
                                                           base_calibrate_fn=get_calibrate_fn(
                                                               method_config=method_config.base_calibrator),
                                                           method_config=method_config,
                                                           seed=seed,
                                                           cfg=cfg)
    else:
        calibrate_fn = get_calibrate_fn(method_config=method_config)
        return calibrate_fn(method_name=method_config.name,
                            train_logits=train_logits,
                            train_labels=train_labels,
                            test_logits=test_test_logits)