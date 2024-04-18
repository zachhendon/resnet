# ResNet

This project builds and trains [ResNet](https://arxiv.org/abs/1512.03385) models from scratch in Pytorch on the CIFAR-10 dataset.


## Run Locally

Clone the project


```
git clone https://github.com/zachhendon/resnet
cd resnet
```

Install dependencies with conda

```
conda env create -n [environment_name] -f environment.yml
conda activate [environment_name]
```


## Training

There are two ResNet variations available to train - resnet18 and resnet34. To train one of these models, run 
```
python3 train.py -m [resnet18|resnet34]
```
To see additional training options, use the `-h` flag.

As the model trains, loss and accuracy metrics are written to Tensorboard. To view these graphs, open the link created by running
```
tensorboard --logdir models
```


When training finishes, the program will save your trained model in the [models](https://github.com/zachhendon/resnet/tree/main/models) directory. 
## Testing

To view the results of a trained model, first find the name of the model's directory. You can find this by looking in the models directory, but it is also printed during training. Then, run
```
python3 test.py -m [model_type] -f [model_directory]

# example
python3 test.py -m resnet18 -f resnet18_20240417_232000
```


The program will then output the accuracy of the trained model on the testing dataset.
## Accuracy

Two models, using the default setup in [train.py](https://github.com/zachhendon/resnet/blob/main/train.py), have already been trained. Their testing accuracy is listed below.

| Model    | Accuracy |
|----------|----------|
| ResNet18 | 93.50%   |
| ResNet34 | 93.29%   |
