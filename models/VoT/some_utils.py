import pylab
import skimage.data
import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision
import cv2

def matplotlib_imshow(img, one_channel=False,title=""):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy()

    fig, ax = plt.subplots()
    if len(title)>0:
        fig.suptitle(title, fontsize=12)

    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_each_image(res,nSub=1,pic_no=-1,picks=None):
    if res.is_cuda:
        res = res.cpu()

    if len(res.shape) == 4:
        imgs = torch.squeeze(res[0,...])
    else:
        imgs = res
    assert(len(imgs.shape) == 3)
    if picks is None:
        picks = list(range(imgs.size(0)))
    for c in picks:
        if nSub>1:
            continue
        img = imgs[c,...]
        img = img.detach().numpy()
        if False:
            img_grid = torchvision.utils.make_grid(imgs)
            print (img_grid.shape)
            matplotlib_imshow(img_grid)
        else:
            pylab.title(f"ID={c} SHAPE={img.shape}")
            fig = pylab.imshow(img)        
            pylab.show()  

def show_tensors(tensors, nr_=16, pad_=10,title=""):
    if tensors.is_cuda:
        tensors = tensors.cpu()
    assert(len(tensors.shape) == 4)
    img_grid = torchvision.utils.make_grid(tensors,nrow=nr_, padding=pad_)
    print (img_grid.shape)
    matplotlib_imshow(img_grid,title=title)

def unitwise_norm(x,axis=None):
    """Compute norms of each output unit separately, also for linear layers."""
    if len(torch.squeeze(x).shape) <= 1:  # Scalars and vectors
        axis = None
        keepdims = False
        return torch.norm(x)
    elif len(x.shape) in [2, 3]:  # Linear layers of shape IO or multihead linear
        # axis = 0
        # axis = 1
        keepdims = True
    elif len(x.shape) == 4:  # Conv kernels of shape HWIO
        if axis is None:
            axis = [0, 1, 2,]
        keepdims = True
    else:
        raise ValueError(f'Got a parameter with shape not in [1, 2, 4]! {x}')
    return torch.sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5

def clip_grad_rc(grad,W,row_major=False,eps = 1.e-3,clip=0.02):   
    #   adaptive_grad_clip
    if len(grad.shape)==2:
        nR,nC = grad.shape
    axis = 1 if row_major else 0
    g_norm = unitwise_norm(grad,axis=axis)
    W_norm = unitwise_norm(W,axis=axis)
    assert(g_norm.shape==W_norm.shape)
    W_norm[W_norm<eps] = eps
    # clipped_grad = grad * (W_norm / g_norm)       
    s = torch.squeeze(clip*W_norm / (g_norm+1.0e-6))     
    s = torch.clamp(s, max=1)
    if len(grad.shape)==1 or s.numel()==nC:       #nC                
        grad = grad*s                
    else:                   #nR           
        grad = torch.einsum('rc,r->rc', grad, s)
    return grad

def random_erase(p, area_ratio_range, min_aspect_ratio, max_attempt=20,value=0):
    sl, sh = area_ratio_range
    rl, rh = min_aspect_ratio, 1. / min_aspect_ratio

    def _random_erase(image): 
        if np.random.random() > p:
            return np.asarray(image).copy()

        image = np.asarray(image).copy()
        nC,img_h,img_w = image.shape[-3:]
        # h, w = image.shape[:2]
        image_area = img_h * img_w

        for i in range(max_attempt):
            mask_area = np.random.uniform(sl, sh) * image_area
            aspect_ratio = np.random.uniform(rl, rh)
            mask_h = int(np.sqrt(mask_area * aspect_ratio))
            mask_w = int(np.sqrt(mask_area / aspect_ratio))

            if mask_w < img_w and mask_h < img_h:
                x0 = np.random.randint(0, img_w - mask_w)
                y0 = np.random.randint(0, img_h - mask_h)
                x1 = x0 + mask_w
                y1 = y0 + mask_h
                image[i%nC,y0:y1, x0:x1] = np.random.uniform(0, 1)
                # 0

        return image

    return _random_erase