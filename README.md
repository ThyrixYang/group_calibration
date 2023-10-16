Code and Appendix of Beyond Probability Partitions: Calibrating Neural Networks with Semantic Aware Grouping (NeurIPS 2023).

## Methods

All the available methods can be found in conf/method

## Datasets

All the implemented datasets can be found in conf/data

We uploaded CIFAR10-Reset152 dataset in datasets/ for reproduction.
To run with this dataset, the path in conf/cifar10_resnet152.yaml should be
modified accordingly.

## Run

To run experiments, run "python main.py +method=method_name +data=data_name", for example,

```bash
python main.py +method=group_calibration_combine_ts +data=cifar10_resnet152
python main.py +method=group_calibration_combine_ets +data=cifar10_resnet152
python main.py +method=histogram_binning +data=cifar10_resnet152
python main.py +method=isotonic_regression +data=cifar10_resnet152
python main.py +method=temp_scaling +data=cifar10_resnet152
python main.py +method=ets +data=cifar10_resnet152
```

To verity the performance without calibration, run with method=none:
```bash
python main.py +method=none +data=cifar10_resnet152
```

And the results will be printed.
