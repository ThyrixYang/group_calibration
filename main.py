import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import time

from data import load_data
from utils import set_seed, gather_metrics
from methods import calibrate
from evaluate import evaluate

def _main(cfg):
    logging.info("config: {}\n===========\n".format(OmegaConf.to_yaml(cfg)))
    seeds = cfg.seeds

    start_time = time.time()
    metrics = []
    for seed in seeds:
        logging.info("Running seed: {}".format(seed))
        
                
        val_data, test_train_data, test_test_data = load_data(data_config=cfg.data,
                                                            seed=seed)
        set_seed(seed)
        
        calibrated_test_test = calibrate(method_config=cfg.method,
                                        val_data=val_data,
                                        test_train_data=test_train_data,
                                        test_test_data=test_test_data,
                                        seed=seed,
                                        cfg=cfg)
        
        _metrics = evaluate(y=test_test_data["labels"],
                            num_classes=test_test_data["logits"].shape[1],
                            n_bins=15,
                            pred_logits=calibrated_test_test.get(
            "logits", None),
            pred_prob=calibrated_test_test.get("prob", None)
        )
        logging.info("Metrics: {}".format(_metrics))
        _results = (seed, _metrics)
            
        metrics.append(_results)
    metric_stats, metrics = gather_metrics(metrics)
    logging.info("Metrics stats: {}".format(metric_stats))


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig) -> None:
    _main(cfg)

if __name__ == "__main__":
    main()