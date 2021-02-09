tensorboard --logdir=/home/cys/net-help/kfac_distribute/logs
tensorboard --logdir=G:\DeepFormer\logs
clear && python ./examples/pytorch_cifar10_resnet.py --kfac-update-freq=-10 --damping=0.003 --base-lr=0.1 --batch-size=128 --model=vit