from __future__ import print_function
import argparse
import time
import os
import sys
import datetime
import math
from distutils.version import LooseVersion
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms, models
import torch.utils.data.distributed

from torchsummary import summary
import cifar_resnet as resnet
# import horovod.torch as hvd
from tqdm import tqdm
from utils import *
from qhoptim.pyt import QHAdam

import sys
root_path = "G:/DeepFormer/"          #"/home/cys/transformer/DeepFormer/"  
sys.path.append(root_path)

import kfac     #export PYTHONPATH=$PYTHONPATH:/home/cys/net-help/kfac_distribute/
from bert_image import *
from vit_pytorch import ViT
from torchvision.models import resnet50
from vit_pytorch.distill import DistillableViT, DistillWrapper
from DeepGraph import build_graph
#   python pytorch_cifar10_resnet.py --kfac-update-freq=-10 --damping=0.003 --base-lr=0.1 --model=Jaggi 
#   python pytorch_cifar10_resnet.py --kfac-update-freq=10 --damping=0.003 --base-lr=0.1 --batch-size=320 --model=Jaggi --gradient_clip=agc --self_attention=gabor 
STEP_FIRST = LooseVersion(torch.__version__) < LooseVersion('1.1.0')
datas_name = "cifar10"
if datas_name == "cifar100":
    datas = datasets.CIFAR100
    nClass = 100
    # IMAGE_W,IMAGE_H=32,32
    IMAGE_W,IMAGE_H=64,64
else:
    datas = datasets.CIFAR10
    nClass = 10
    # IMAGE_W,IMAGE_H=32,32
    IMAGE_W,IMAGE_H=64,64
# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Example')
parser.add_argument('--model', type=str, default='resnet32',
                    help='ResNet model to use [20, 32, 56]')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='WE',
                    help='number of warmup epochs (default: 5)')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')

# Optimizer Parameters
parser.add_argument('--base-lr', type=float, default=0.1, metavar='LR',
                    help='base learning rate (default: 0.1)')
parser.add_argument('--lr-decay', nargs='+', type=int, default=[100, 150],
                    help='epoch intervals to decay lr')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                    help='SGD weight decay (default: 5e-4)')

# KFAC Parameters
parser.add_argument('--kfac-update-freq', type=int, default=10,
                    help='iters between kfac inv ops (0 for no kfac updates) (default: 10)')
parser.add_argument('--kfac-cov-update-freq', type=int, default=1,
                    help='iters between kfac cov ops (default: 1)')
parser.add_argument('--kfac-update-freq-alpha', type=float, default=10,
                    help='KFAC update freq multiplier (default: 10)')
parser.add_argument('--kfac-update-freq-schedule', nargs='+', type=int, default=None,
                    help='KFAC update freq schedule (default None)')
parser.add_argument('--stat-decay', type=float, default=0.95,
                    help='Alpha value for covariance accumulation (default: 0.95)')
parser.add_argument('--damping', type=float, default=0.003,
                    help='KFAC damping factor (defaultL 0.003)')
parser.add_argument('--damping-alpha', type=float, default=0.5,
                    help='KFAC damping decay factor (default: 0.5)')
parser.add_argument('--damping-schedule', nargs='+', type=int, default=None,
                    help='KFAC damping decay schedule (default None)')

parser.add_argument('--kl-clip', type=float, default=0.001,help='KL clip (default: 0.001)')
parser.add_argument('--gradient_clip', type=str, default="agc",help='KL clip (default: 0.001)')
parser.add_argument('--self_attention', type=str, default="gaussian",help='KL clip (default: 0.001)')

parser.add_argument('--diag-blocks', type=int, default=1,
                    help='Number of blocks to approx layer factor with (default: 1)')
parser.add_argument('--diag-warmup', type=int, default=5,
                    help='Epoch to start diag block approximation at (default: 5)')
parser.add_argument('--distribute-layer-factors', action='store_true', default=False,
                    help='Compute A and G for a single layer on different workers')

# Other Parameters
parser.add_argument('--log-dir', default=f'./logs/{datas_name}/',help='TensorBoard log directory')
#/home/cys/Downloads/{datas_name}/
parser.add_argument('--dir', type=str, default=f'C:/Users/cys/Downloads/{datas_name}/', metavar='D',help='directory to download dataset to')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--positional_encoding', type=str, default=f'')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
# args.stat_decay = 0       #nealy same 
isHVD = "horovod" in sys.modules
verbose = True
device = 0
sGPU = ""
download = True
num_replicas=1
rank=0

class DeepLogger(SummaryWriter):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = -1
        self.batch_idx = -1

    def isPlot(self):
        return self.batch_idx==2
         
def clip_grad(model,eps = 1.e-3,clip=0.02,method="agc"):   
    known_modules = {'Linear'} 
    for module in model.modules():
        classname = module.__class__.__name__   
        if classname not in known_modules:
            continue
        if classname == 'Conv2d':
            assert(False)
            grad = None            
        elif classname == 'BertLayerNorm':
            grad = None
        else:
            grad = module.weight.grad.data       
            W = module.weight.data 

        #   adaptive_grad_clip
        assert len(grad.shape)==2
        nR,nC = grad.shape
        axis = 1 if nR>nC else 0
        g_norm = unitwise_norm(grad,axis=axis)
        W_norm = unitwise_norm(W,axis=axis)
        W_norm[W_norm<eps] = eps
        # clipped_grad = grad * (W_norm / g_norm)       
        s = torch.squeeze(clip*W_norm / (g_norm+1.0e-6))     
        s = torch.clamp(s, max=1)
        if s.numel()==nC:       #nC                
            grad = grad*s                
        else:                   #nR           
            grad = torch.einsum('rc,r->rc', grad, s)
        module.weight.grad.data.copy_(grad)

        if module.bias is not None:
            pass    #grad = torch.cat([grad, module.bias.grad.data.view(-1, 1)], 1)

def train(epoch):
    log_writer.epoch = epoch
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    
    if STEP_FIRST:
        for scheduler in lr_scheduler:
            scheduler.step()
        if use_kfac:
            kfac_param_scheduler.step(epoch)
    x_info = ""
    t0 = time.time()
    batch_size = config.batch_size
    report_frequency = (int)(len(train_loader)/100.0+0.5)
    # with tqdm(total=len(train_loader), position=0, leave=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', 
    #           desc='Epoch {:3d}/{:3d}'.format(epoch + 1, config.epochs),
    #           disable=not verbose) as t:      always new line in WINDOWS!!!
    for batch_idx, (data, target) in enumerate(train_loader):
        log_writer.batch_idx = batch_idx
        if args.cuda:                
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        for i in range(0, len(data),batch_size ): 
            data_batch = data[i:i + batch_size]
            target_batch = target[i:i + batch_size]
            if model_name == "distiller":
                loss = distiller(data_batch, target_batch)
                output = model(data_batch)
            else:
                output = model(data_batch)
                if model_name == "vit":
                    output = F.log_softmax(output, dim=1)
                    loss = F.nll_loss(output, target_batch)
                else:
                    loss = criterion(output, target_batch)
            with torch.no_grad():
                train_loss.update(loss)
                train_accuracy.update(accuracy(output, target_batch))
            loss.div_(math.ceil(float(len(data)) / batch_size))
            loss.backward()
        if isHVD:
            optimizer.synchronize()
        acc = train_accuracy.avg.item()
        if use_kfac:
            preconditioner.step(epoch=epoch,accuracy=acc)                
            x_info = preconditioner.dump(epoch + 1,log_writer)
        if isHVD:
            with optimizer.skip_synchronize():
                optimizer.step()
        else:     
            clip_grad(model)   #need this if LayerNormal=identity    
            optimizer.step()
            
            

        lr_opt = optimizer.param_groups[0]['lr']
        # t.set_postfix_str("lr={:.6f}, loss: {:.4f}, acc: {:.2f}%, {},T={:.1f}".format(lr_opt,train_loss.avg.item(), 100*acc,x_info,time.time()-t0))
        # t.update(1)
        if batch_idx % report_frequency == 0:
            print(f"\rEpoch {epoch + 1:3d}/{config.epochs:3d}\t{batch_idx}/{len(train_loader)} \tlr={lr_opt:.3f}, loss: {train_loss.avg.item():.4f}, acc: {100*acc:.2f}%, {x_info},T={time.time()-t0:.1f}",end="")
    print(f"")

    if not STEP_FIRST:
        for scheduler in lr_scheduler:
            scheduler.step()
            # print(f'\t{epoch} scheduler.get_last_lr={scheduler.get_last_lr()[0]:.6f}')
        if use_kfac:
            kfac_param_scheduler.step(epoch)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)

def test(epoch):
    model.eval()
    test_loss = Metric('val_loss')
    test_accuracy = Metric('val_accuracy')
    
    with tqdm(total=len(test_loader), bar_format='{l_bar}{bar:10}|{postfix}',
              desc='             '.format(epoch + 1, config.epochs),
              disable=not verbose) as t:
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                if model_name == "vit":
                    output = F.log_softmax(output, dim=1)
                    test_loss.update(F.nll_loss(output, target))
                else:
                    test_loss.update(criterion(output, target))
                test_accuracy.update(accuracy(output, target))
                
                t.update(1)
                if i + 1 == len(test_loader):
                    t.set_postfix_str("\b\b test_loss: {:.4f}, test_acc: {:.2f}%".format(
                            test_loss.avg.item(), 100*test_accuracy.avg.item()),
                            refresh=False)

    if log_writer:
        log_writer.add_scalar('test/loss', test_loss.avg, epoch)
        log_writer.add_scalar('test/accuracy', test_accuracy.avg, epoch)

if __name__ == "__main__":
    if isHVD:      #So strange!!! this would affect by TensorBoard
        print(f"hvd={hvd.local_rank()} size={hvd.size()}")
        verbose = True if hvd.rank() == 0 else False
        device = hvd.local_rank()
        sGPU = f"gpu*{hvd.size()}" if hvd.size()>1 else ""
        download = True if hvd.local_rank() == 0 else False
        if not download: hvd.allreduce(torch.tensor(1), name="barrier")
        num_replicas=hvd.size()
        rank=hvd.rank()
        args.base_lr = args.base_lr * hvd.size()
    if args.cuda:
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    kfac.seed_everything(42)
    sFac = f"KFAC_" if args.kfac_update_freq>0 else ""
# args.log_dir = os.path.join(args.log_dir, "__{}_{}_{}_{}".format(args.model, sFac, sGPU,datetime.datetime.now().strftime('%m-%d_%H-%M')))
    print(args)
    # os.makedirs(args.log_dir, exist_ok=True)
    # log_writer = SummaryWriter(args.log_dir) if verbose else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize([IMAGE_W,IMAGE_H]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),        
        # transforms.RandomErasing(),      #p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
        # random_erase( p=0.5, area_ratio_range=(0.02, 0.3), min_aspect_ratio=0.1, max_attempt=6),
        # transforms.ToTensor(),
        ])
    transform_test = transforms.Compose([
        transforms.Resize([IMAGE_W,IMAGE_H]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        ])

    train_dataset = datas(root=args.dir, train=True,download=download, transform=transform_train)
    test_dataset = datas(root=args.dir, train=False,download=download, transform=transform_test)

    config = args
    args.log_dir = os.path.join(f"{root_path}/logs/", "{}/__{}_{}_{}_{}".format(datas_name,args.model, sFac, sGPU,datetime.datetime.now().strftime('%m-%d_%H-%M')))
    os.makedirs(args.log_dir, exist_ok=True)
    # log_writer = SummaryWriter(args.log_dir) if verbose else None
    log_writer = DeepLogger(args.log_dir) if verbose else None
    model_name = args.model.lower()

    if model_name == "resnet20":
        model = resnet.resnet20()
    elif model_name == "resnet32":
        model = resnet.resnet32()
    elif model_name == "resnet44":
        model = resnet.resnet44()
    elif model_name == "resnet56":
        model = resnet.resnet56()
    elif model_name == "resnet110":
        model = resnet.resnet110()
    elif model_name == "vit":
        #hidden=256,very BAD!!!
        #lr=0.001 nearly same
        model = ViT(image_size = 32,patch_size = 4,num_classes = nClass,dim = 21,depth = 6,heads = 3,ff_hidden = 128,dropout = 0,emb_dropout = 0.1)    
        # model = ImageTransformer(image_size=32, patch_size=4, num_classes=nClass, channels=3,dim=64, depth=6, heads=8, mlp_dim=128)          #
        # model = ViT(image_size = 256,patch_size = 32,num_classes = 1000,dim = 1024,depth = 6,eads = 16,mlp_dim = 2048,dropout = 0.1,emb_dropout = 0.1)
        #24 overfit
    elif model_name == "distiller":
        teacher = resnet50(pretrained = True)
        teacher.cuda()
        model = DistillableViT(image_size = 32,patch_size = 4,num_classes = nClass,dim = 64,depth = 6,heads = 8,mlp_dim = 128,dropout = 0.1,emb_dropout = 0.1)
        distiller = DistillWrapper(student = model,teacher = teacher,temperature = 3,alpha = 0.5)
    elif model_name == "lamlay":
        args.batch_size = 128;          args.weight_decay=0.0001
        # model = lambda_resnet26()
        model = LambdaResNet18()
        args.log_dir=f"./logs/lamlay/"
    elif model_name == "jaggi":
        BertImage_config['use_attention'] = config.self_attention
        BertImage_config['gradient_clip'] = config.gradient_clip
        BertImage_config['INPUT_W'] = (int)(IMAGE_W / BertImage_config['pooling_concatenate_size'])
        BertImage_config['INPUT_H'] = (int)(IMAGE_H / BertImage_config['pooling_concatenate_size'])

        BertImage_config['logger'] = log_writer
        BertImage_config['positional_encoding'] = config.positional_encoding
        model = BertImage(BertImage_config, num_classes=nClass)
        config = Namespace(**BertImage_config)
        # args.log_dir=f"/home/cys/net-help/kfac_distribute/logs/Jaggi/"
    
    # config.log_writer = log_writer
    print(model)
    # g = build_graph(model, torch.zeros([1, 3, 64, 64]),path="./2.pdf")         # transforms=transforms

    batch_size = config.batch_size
    if download and isHVD: hvd.allreduce(torch.tensor(1), name="barrier")
    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=num_replicas, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=batch_size * args.batches_per_allreduce, 
            sampler=train_sampler, **kwargs)

    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=num_replicas, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
            batch_size=batch_size, sampler=test_sampler, **kwargs)

    if args.cuda:
        model.cuda()
    # if verbose:    summary(model, (3, 32, 32))

    criterion = nn.CrossEntropyLoss()
    use_kfac = True if args.kfac_update_freq > 0 else False

    # if use_kfac:    args.base_lr = 0.003     #0.003 for vit
    lr_scheduler=[]
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,weight_decay=args.weight_decay)
    if model_name == "vit" or model_name == "distiller":
        optimizer = optim.Adam(model.parameters(), lr=0.003)        #QHAdam is nearly same as adam, much better than SGD
        # optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=args.momentum,weight_decay=args.weight_decay)
    if model_name == "jaggi":
        optimizer, lrs = Jaggi_get_optimizer(train_loader,model.named_parameters(),BertImage_config)
        # lr_scheduler.append(lrs)

    if use_kfac:
        kfac_core = kfac.CG_KFAC if False else kfac.KFAC
        # preconditioner = kfac.KFAC(model, lr=args.base_lr, factor_decay=args.stat_decay, 
        #                            damping=args.damping, kl_clip=args.kl_clip, 
        #                            fac_update_freq=args.kfac_cov_update_freq, 
        #                            kfac_update_freq=args.kfac_update_freq,
        #                            diag_blocks=args.diag_blocks,
        #                            diag_warmup=args.diag_warmup,
        #                            distribute_layer_factors=args.distribute_layer_factors)
        preconditioner = kfac_core(model, lr=args.base_lr, factor_decay=args.stat_decay, 
                                damping=args.damping, kl_clip=args.kl_clip, 
                                gradient_clip = config.gradient_clip,
                                fac_update_freq=args.kfac_cov_update_freq, 
                                kfac_update_freq=args.kfac_update_freq,
                                diag_blocks=args.diag_blocks,
                                diag_warmup=args.diag_warmup,
                                distribute_layer_factors=args.distribute_layer_factors)
        kfac_param_scheduler = kfac.KFACParamScheduler(preconditioner,
                damping_alpha=args.damping_alpha,
                damping_schedule=args.damping_schedule,
                update_freq_alpha=args.kfac_update_freq_alpha,
                update_freq_schedule=args.kfac_update_freq_schedule)
    else:
        preconditioner = None

    print(f"======== optimizer={optimizer}\n\n======== MODEL={model.name_()}\n======== preconditioner={preconditioner}")
    # KFAC guarentees grads are equal across ranks before opt.step() is called
    # so if we do not use kfac we need to wrap the optimizer with horovodcon
    if isHVD:
        compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(optimizer, 
                                            named_parameters=model.named_parameters(),
                                            compression=compression,
                                            op=hvd.Average,
                                            backward_passes_per_step=args.batches_per_allreduce)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    if len(lr_scheduler)==0:
        #...5...[100, 150]...
        lrs = create_lr_schedule(num_replicas, args.warmup_epochs, args.lr_decay)
        lr_scheduler = [LambdaLR(optimizer, lrs)]
    if use_kfac:
        lr_scheduler.append(LambdaLR(preconditioner, lrs))
    for ls in lr_scheduler:
        print(f"======== lr_scheduler={ls.state_dict()}")
    start = time.time()

    for epoch in range(config.epochs):
        train(epoch)
        test(epoch)

    if verbose:
        print("\nTraining time:", str(datetime.timedelta(seconds=time.time() - start)))

    print(f"======== args={args}\n")
    print(f"======== config={config}\n")
    print(f"======== B={optimizer}\n======== preconditioner={preconditioner}")
    print(f"======== optimizer={optimizer}\n======== preconditioner={preconditioner}")