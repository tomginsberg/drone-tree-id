# Dense Canopy Tree Crown Instance Segmentation

A FYBR Solutions Inc. engine

<div align="center">
  <img src="docs/home.png"/>
</div>

## Framework Structure
* `configs/` - definition of config files, which defines architecture, pretrained model to load, hyperparameters...
* `lib/` - libraries and dependencies
* `deepent/` - our library
    * `data` - module for data preprocessing and processing for training and evaluation
    * `modelling` - model architectures
* `output/` - files generated by framework such as models, logs, summaries, checkpoints ...
* `notebooks/` - documentation logbooks
* `docs/` - model zoo descrioptions
* `setup` - setup script
* `train` - train script for Leonhard
* `train_net.py` - trianing procedure
* `vis.py` - a tool for creating visualizations

## Environments

### Leonhard

Ensure you are connected to the ETH network.
```
ssh [your_ETH_alias]@login.leonhard.ethz.ch
```
Remember to load modules:
```
module load gcc/6.3.0 python_gpu/3.7.4
```
And add `--user` arg to `pip install` commands on Leonhard. 
### AWS

Remember to load the pytorch build:
```
source activate pytorch_p36
```

### Setup

1. Clone repo:
```
cd ~
git clone https://github.com/roeetal/drone-tree-id.git
```
2. run setup script (for leonhard modify pip install to take `--user` argument):
```
cd ~/drone-tree-id
chmod +x setup
./setup
```

## Benchmarking
To benchmark training or evalutation sessions:

## Training

To train the model, run the training script with appropriate flags:
* `--config-file`: the config file to load
* `OUTPUT_DIR`: the output path directory, for example: output/<name of experiment>
For example:
```
python train_net.py --config-file configs/deepent_rcnn_R_101_FPN.yaml OUTPUT_DIR path/to/output/dir
```

### Leonhard

You can run a batch training by compiling and executing the `train` script

## Evaluation

To evaluation the model, run the training script with appropriate flags:
* `--config-file`: the config file to load
* `MODEL.WEIGHTS`: the weight file to load
For example:
```
python train_net.py --config-file /home/ubuntu/drone-tree-id/configs/deepent_rcnn_R_50_FPN.yaml --eval-only MODEL.WEIGHTS /home/ubuntu/drone-tree-id/output/model_0034999.pth
```

## Visualization

To create a 6x2 sample visualization of model inference, run the following with the appropriate arguments:
* `--config-file`: the config file to load
* `--model`: the weight file to load`
* `--dataset`: name of the dataset
* `--threshold`: confidence threshold
* `--samples`: number of sample visualizations to produce
* `--output`: path to store visualizations
```
python vis.py --threshold 0.5 --config-file configs/deepent_rcnn_R_50_FPN.yaml --model output/baseline_17_01_2019/model_0019999.pth --samples 1 --dataset SAL_test --output output/baseline_17_01_2019/vis/ --type comparison
```
and then transfer
```
scp -r -i ~/.ssh/vm.pem ubuntu@ec2-54-226-164-33.compute-1.amazonaws.com:/home/ubuntu/drone-tree-id/output/baseline_17_01_2019/vis Desktop/vis
```
