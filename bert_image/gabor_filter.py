import torch
import torch.nn as nn
import math
import numbers
import torch
import numpy
from torch import nn
from torch.nn import functional as F
import sys
from .some_utils import show_tensors

class GaborFilters(nn.Module):
    def __init__(self, 
        in_channels, 
        n_sigmas = 3,
        n_lambdas = 4,
        n_gammas = 1,
        n_thetas = 7,
        kernel_radius=15,
        rotation_invariant=True
    ):
        super().__init__()
        self.in_channels = in_channels
        kernel_size = kernel_radius*2 + 1
        self.kernel_size = kernel_size
        self.kernel_radius = kernel_radius
        self.n_thetas = n_thetas
        self.rotation_invariant = rotation_invariant
        def make_param(in_channels, values, requires_grad=True, dtype=None):
            if dtype is None:
                dtype = 'float32'
            values = numpy.require(values, dtype=dtype)
            n = in_channels * len(values)
            data=torch.from_numpy(values).view(1,-1)
            data = data.repeat(in_channels, 1)
            return torch.nn.Parameter(data=data, requires_grad=requires_grad)


        # build all learnable parameters
        self.sigmas = make_param(in_channels, 2**numpy.arange(n_sigmas)*2)
        self.lambdas = make_param(in_channels, 2**numpy.arange(n_lambdas)*4.0)
        self.gammas = make_param(in_channels, numpy.ones(n_gammas)*0.5)
        self.psis = make_param(in_channels, numpy.array([0, math.pi/2.0]))

        print(len(self.sigmas))


        thetas = numpy.linspace(0.0, 2.0*math.pi, num=n_thetas, endpoint=False)
        thetas = torch.from_numpy(thetas).float()
        self.register_buffer('thetas', thetas)

        indices = torch.arange(kernel_size, dtype=torch.float32) -  (kernel_size - 1)/2
        self.register_buffer('indices', indices)


        # number of channels after the conv
        self._n_channels_post_conv = self.in_channels * self.sigmas.shape[1] * \
                                     self.lambdas.shape[1] * self.gammas.shape[1] * \
                                     self.psis.shape[1] * self.thetas.shape[0] 


    def make_gabor_filters(self):
        sigmas=self.sigmas
        lambdas=self.lambdas
        gammas=self.gammas
        psis=self.psis
        thetas=self.thetas      #7 different angles
        y=self.indices
        x=self.indices

        in_channels = sigmas.shape[0]
        assert in_channels == lambdas.shape[0]
        assert in_channels == gammas.shape[0]

        kernel_size = y.shape[0], x.shape[0]



        sigmas  = sigmas.view (in_channels, sigmas.shape[1],1, 1, 1, 1, 1, 1)
        lambdas = lambdas.view(in_channels, 1, lambdas.shape[1],1, 1, 1, 1, 1)
        gammas  = gammas.view (in_channels, 1, 1, gammas.shape[1], 1, 1, 1, 1)
        psis    = psis.view (in_channels, 1, 1, 1, psis.shape[1], 1, 1, 1)

        thetas  = thetas.view(1,1, 1, 1, 1, thetas.shape[0], 1, 1)
        y       = y.view(1,1, 1, 1, 1, 1, y.shape[0], 1)
        x       = x.view(1,1, 1, 1, 1, 1, 1, x.shape[0])

        sigma_x = sigmas
        sigma_y = sigmas / gammas

        sin_t = torch.sin(thetas)
        cos_t = torch.cos(thetas)
        y_theta = -x * sin_t + y * cos_t
        #   [channel,sigma,lambda,gamms,psis,  xita,x,y] => 1, 1, 1, 1, 1, 7, 31, 31
        x_theta =  x * cos_t + y * sin_t     
        K0,K1=kernel_size[0], kernel_size[1]
        if True:
            gaussian = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))
            # show_each_channel(gaussian.view(-1,K0,K1),picks=[0,1,2,8,9,-2,-1])
            print(f"gaussian=={gaussian.shape} => {gaussian.view(-1,K0,K1).shape}")
            #   [channel,..., psi, xita,x,y] => [3, 1, 4, 1, 2, 7, 31, 31]
            wave = torch.cos(2.0 * math.pi  * x_theta / lambdas + psis)
            print(f"wave={wave.shape} => {wave.view(-1,K0,K1).shape}")
            # show_each_channel(wave.view(-1,kernel_size[0], kernel_size[1]))
            gb = gaussian*wave
            # show_each_channel(gb.view(-1,kernel_size[0], kernel_size[1]),picks=[0,1,2,8,9,-2,-1])
        else:
            gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) \
                * torch.cos(2.0 * math.pi  * x_theta / lambdas + psis)

        print(f"gb={gb.shape} => {gb.view(-1,K0,K1).shape}")
        gb = gb.view(-1,kernel_size[0], kernel_size[1])
        
        # show_each_channel(gb)
        return gb


    def forward(self, x):
        batch_size = x.size(0)
        sy = x.size(2)
        sx = x.size(3)  
        gb = self.make_gabor_filters()

        assert gb.shape[0] == self._n_channels_post_conv
        assert gb.shape[1] == self.kernel_size
        assert gb.shape[2] == self.kernel_size
        gb = gb.view(self._n_channels_post_conv,1,self.kernel_size,self.kernel_size)

        res = nn.functional.conv2d(input=x, weight=gb,
            padding=self.kernel_radius, groups=self.in_channels)
       
        
        if self.rotation_invariant:
            res = res.view(batch_size, self.in_channels, -1, self.n_thetas,sy, sx)
            res,_ = res.max(dim=3)

        res = res.view(batch_size, -1,sy, sx)


        return res

class GaborConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
    ):
        super().__init__()

        self.is_calculated = False

        self.conv_layer = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.kernel_size = self.conv_layer.kernel_size

        # small addition to avoid division by zero
        self.delta = 1e-3

        # freq, theta, sigma are set up according to S. Meshgini,
        # A. Aghagolzadeh and H. Seyedarabi, "Face recognition using
        # Gabor filter bank, kernel principal component analysis
        # and support vector machine"
        self.freq = Parameter(
            (math.pi / 2)
            * math.sqrt(2)
            ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor),
            requires_grad=True,
        )
        self.theta = Parameter(
            (math.pi / 8)
            * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor),
            requires_grad=True,
        )
        self.sigma = Parameter(math.pi / self.freq, requires_grad=True)
        self.psi = Parameter(
            math.pi * torch.rand(out_channels, in_channels), requires_grad=True
        )

        self.x0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0], requires_grad=False
        )
        self.y0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0], requires_grad=False
        )

        self.y, self.x = torch.meshgrid(
            [
                torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),
                torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1]),
            ]
        )
        self.y = Parameter(self.y)
        self.x = Parameter(self.x)

        self.weight = Parameter(
            torch.empty(self.conv_layer.weight.shape, requires_grad=True),
            requires_grad=True,
        )

        self.register_parameter("freq", self.freq)
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("psi", self.psi)
        self.register_parameter("x_shape", self.x0)
        self.register_parameter("y_shape", self.y0)
        self.register_parameter("y_grid", self.y)
        self.register_parameter("x_grid", self.x)
        self.register_parameter("weight", self.weight)

    def forward(self, input_tensor):
        if self.training:
            self.calculate_weights()
            self.is_calculated = False
        if not self.training:
            if not self.is_calculated:
                self.calculate_weights()
                self.is_calculated = True
        return self.conv_layer(input_tensor)

    def calculate_weights(self):
        for i in range(self.conv_layer.out_channels):
            for j in range(self.conv_layer.in_channels):
                sigma = self.sigma[i, j].expand_as(self.y)
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)

                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

                g = torch.exp(
                    -0.5 * ((rotx ** 2 + roty ** 2) / (sigma + self.delta) ** 2)
                )
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * math.pi * sigma ** 2)
                self.conv_layer.weight.data[i, j] = g

class QKV_(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        #mini batch多句话得长度并不一致,需要按照最大得长度对短句子进行补全，也就是padding零，mask起来，填充一个负无穷（-1e9这样得数值），这样计算就可以为0了，等于把计算遮挡住。
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        # p_attn = entmax15(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class QKV_MultiHead(nn.Module):
    class QKV_2D(nn.Module):
        def forward(self, query, key, value, mask=None, dropout=None):
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
            s = math.sqrt(query.size(-1))
            scores = torch.einsum('bijd,bijd->bij', query, key) / s
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            p_attn = F.softmax(scores, dim=-1)
            # p_attn = entmax15(scores, dim=-1)

            if dropout is not None:
                p_attn = dropout(p_attn)
            qkv = torch.einsum('bij,bijd->bijd', p_attn, value)
            # return torch.matmul(p_attn, value), p_attn
            return qkv,p_attn
            
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        # assert d_model % h == 0

        self.d_k = d_model #// h
        self.h = h

        self.projects =    nn.ModuleList( nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])   for _ in range(self.h))
        #self.projects =     nn.ModuleList( nn.ModuleList([nn.Identity() for _ in range(3)])                 for _ in range(self.h))        #UGLY
        if self.h>1:
            self.output_linear = nn.Linear(d_model*self.h, d_model)
        else:
            pass
        
        self.qkv = self.QKV_2D()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        shape_0,nEle_0 = x.shape,x.numel()
        batch_size = x.size(0)
        arrX = []
        if self.qkv is None:
            x = self.dropout(x)         #Very interesting, why self-attention is so useful?
        else:
            for i in range(self.h):
                query, key, value = [l(x).view(batch_size, -1, 1, self.d_k).transpose(1, 2)for l, x in zip(self.projects[i], (x, x, x))]
                x, attn = self.qkv(query, key, value, mask=None, dropout=self.dropout)
                x = x.transpose(1, 2)
                assert x.numel() == nEle_0
                arrX.append( x.contiguous().view(shape_0) )
        if self.h>1:
            x = torch.stack(arrX,dim=-1)
            x = x.view(shape_0[0],shape_0[1],shape_0[2],-1)          #bEijH => bij(EH=1024)
            x = self.output_linear(x)
            return x
        else:
            return arrX[0]

class GaborSelfAttention(nn.Module):
    def _init_gabor_(self,config,kernel_size,n_lambdas = 4,n_phase=2,n_thetas=7):
        from .some_utils import show_tensors
        indices = torch.arange(kernel_size, dtype=torch.float32) -  (kernel_size - 1)/2
        x,y = indices,indices        
        
        thetas = numpy.linspace(0.0, 2.0*math.pi, num=n_thetas, endpoint=False)
        thetas = torch.from_numpy(thetas).float()
        self.register_buffer('thetas', thetas)                              #[0.0000, 0.8976, 1.7952, 2.6928, 3.5904, 4.4880, 5.3856]
        # self.psis = torch.from_numpy(numpy.array([0, math.pi/2.0]))         #[0,1.57]
        self.psis = torch.from_numpy(math.pi/2.0*numpy.arange(n_phase))
        # self.lambdas = torch.from_numpy(2**numpy.arange(n_lambdas)*4.0)     #[4,8,16,32]
        self.lambdas = torch.from_numpy(2**numpy.arange(n_lambdas)*8.0)     #[4,8,16,32]

        self.lambdas = self.lambdas.view( self.lambdas.shape[0],1, 1, 1, 1)             #wavelength
        self.psis    = self.psis.view (   1, self.psis.shape[0], 1, 1, 1)               #phase
        self.thetas  = self.thetas.view(  1, 1, self.thetas.shape[0],  1,       1)      #orientation
        y       = y.view(       1, 1, 1,            y.shape[0], 1)
        x       = x.view(       1, 1, 1,                1,      x.shape[0])
        sin_t = torch.sin(self.thetas)
        cos_t = torch.cos(self.thetas)
        y_theta = -x * sin_t + y * cos_t
        #   [channel,sigma,lambda,gamms,psis,  xita,x,y] => 1, 1, 1, 1, 1, 7, 31, 31
        self.x_theta =  x * cos_t + y * sin_t
        self.wave = torch.cos(2.0 * math.pi  * self.x_theta / self.lambdas + self.psis)
        print(f"wave={self.wave.shape} = {self.wave.min()} - {self.wave.max()}")
        # show_tensors(self.wave.contiguous().view(-1,1,kernel_size,kernel_size), nr_=n_thetas, pad_=10)
        # print(f"wave={wave.shape} => {wave.view(-1,K0,K1).shape}")
        self.wave = nn.Parameter(self.wave.float())     #self.wave.cuda().float()

    def grid2RPE_5(self,grid,xita=None):      #grid(X,Y) =>   relative position encoding (dX,dY,dX2,dY2,dXdY)
        if xita is not None:
            rotate = torch.tensor( [[math.cos(xita),math.sin(xita)],[-math.sin(xita),math.cos(xita)]] )
            grid = torch.einsum('ijc,cr->ijr', [grid.float(), rotate])      #cr: rotate vector of each codinate
        relative_indices = grid.unsqueeze(0).unsqueeze(0) - grid.unsqueeze(-2).unsqueeze(-2)        #so elegant and hard code
        R = torch.cat([relative_indices, relative_indices ** 2, (relative_indices[..., 0] * relative_indices[..., 1]).unsqueeze(-1)], dim=-1)
        R = R.float()
        return R 

    def __init__(self, config, hidden_in,output_attentions=False, keep_multihead_output=False,title=""):
        super().__init__()
        self.config = config
        self.title = title+"_gabor"
        # self.attention_dropout = nn.Dropout(p=0.1)
        self.guided_filter = None   #SelfGuidedFilter(3,8,8)
        width,height = config.INPUT_W,  config.INPUT_H

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = hidden_in    #config.hidden_size
        # assert config.hidden_size % config.num_attention_heads == 0, "num_attention_heads should divide hidden_size"
        # self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * hidden_in   #config.hidden_size
        self.output_attentions = output_attentions
        self.se_excite = None   #se_basic(self.num_attention_heads,reduction=1)
        self.multiQKV = None    #QKV_MultiHead(1,d_model=self.attention_head_size)        #self.num_attention_heads,self.all_head_size
        # shift of the each gaussian per head
        self.attention_centers = nn.Parameter(
            torch.zeros(self.num_attention_heads, 2).normal_(0.0, config.gaussian_init_mu_std)
        )

        self.isRXitas = True
        self.isSigma = False         #90%=>88%

        # Inverse standart deviation $Sigma^{-1/2}$
        # 2x2 matrix or a scalar per head
        # initialized to noisy identity matrix
        attention_spreads = torch.eye(2).unsqueeze(0).repeat(self.num_attention_heads, 1, 1)
        attention_spreads += torch.zeros_like(attention_spreads).normal_(0, config.gaussian_init_sigma_std)

        if self.isSigma:        #some bugs!
            self.attention_spreads = attention_spreads
            attention_spreads = self.get_heads_target_vectors()
            print(attention_spreads)
            if self.isRXitas:
                self.sigmaLayer = nn.Linear(5,self.num_attention_heads)  
            else:
                self.sigmaLayer = nn.Linear(5,self.num_attention_heads)   
            with torch.no_grad():
                self.sigmaLayer.weight.copy_(attention_spreads)
        else:
            self.attention_spreads = nn.Parameter(attention_spreads)

        self.fc_allhead2hidden = nn.Linear(self.all_head_size,hidden_in ) 

        if not config.attention_gaussian_blur_trick:
            # relative encoding grid (delta_x, delta_y, delta_x**2, delta_y**2, delta_x * delta_y)
            MAX_WIDTH_HEIGHT = 50
            range_ = torch.arange(MAX_WIDTH_HEIGHT)
            grid = torch.cat([t.unsqueeze(-1) for t in torch.meshgrid([range_, range_])], dim=-1)
            if self.isRXitas:
                listR=[]
                for xita in numpy.linspace(0.0, 2.0*math.pi, num=self.num_attention_heads, endpoint=False):
                    listR.append(self.grid2RPE_5(grid,0)[:width,:height,:width,:height,:])
                R_xitas_ = torch.stack(listR)
                self.register_buffer("R_xitas_", R_xitas_)  
            else:
                self.R_xitas_ = None
            if True:
                R = self.grid2RPE_5(grid,math.pi/4)            
                self.register_buffer("R", R)            #parameters which should be saved and restored in the state_dict, but not trained by the optimizer,
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self._init_gabor_(config,kernel_size=8,n_lambdas = 1,n_phase=1,n_thetas=self.num_attention_heads )

    def get_heads_target_vectors(self):
        inv_covariance = torch.einsum('hij,hkj->hik', [self.attention_spreads, self.attention_spreads])
        a, b, c = inv_covariance[:, 0, 0], inv_covariance[:, 0, 1], inv_covariance[:, 1, 1]

        mu_1, mu_2 = self.attention_centers[:, 0], self.attention_centers[:, 1]

        t_h = -1/2 * torch.stack([
            -2*(a*mu_1 + b*mu_2),
            -2*(c*mu_2 + b*mu_1),
            a,
            c,
            2 * b
        ], dim=-1)
        return t_h

    def get_attention_probs(self, width, height):
        """Compute the positional attention for an image of size width x height
        Returns: tensor of attention probabilities (width, height, num_head, width, height)
        """
        if self.isSigma:
            d = 5
            if self.R_xitas_ is not None:  #would CRASH
                R1 = self.R_xitas_.reshape(-1,d).contiguous()
            else:
                R1 = self.R[:width,:height,:width,:height,:].reshape(-1,d).contiguous()
            gaussian_ = self.sigmaLayer(R1)  #R1@(u.permute(1,0))
            gaussian_ = gaussian_.view(width,height,width,height,-1).permute(0,1,4,2,3).contiguous()
        else:
            u = self.get_heads_target_vectors()
        # Compute attention map for each head
            if True:
                gaussian_ = torch.einsum('hijkld,hd->ijhkl', [self.R_xitas_, u])
                # for h in self.config.num_head:
                #     gaussian_ = torch.einsum('ijkld,d->ijkl', [self.R_xita[h], u[h]])
            else:
                gaussian_ = torch.einsum('ijkld,hd->ijhkl', [self.R[:width,:height,:width,:height,:], u])
        # show_tensors(gaussian_[:,:,-1,:,:].contiguous().view(-1,1,width,height), nr_=8, pad_=4)
        # Softmax
        
        if False:
            w = self.wave[:,:,:,:width,:height]
            w = torch.squeeze(self.wave)
            gaussian_ = torch.einsum('ijhkl,hkl->ijhkl', [gaussian_,w])
      
        attention_probs = torch.nn.Softmax(dim=-1)(gaussian_.view(width, height, self.num_attention_heads, -1))
        # attention_probs = entmax15(attention_scores.view(width, height, self.num_attention_heads, -1),dim=-1)
        attention_probs = attention_probs.view(width, height, self.num_attention_heads, width, height)

        return attention_probs

    def forward(self, X, attention_mask, head_mask=None):
        LOG = self.config.logger
        assert len(X.shape) == 4
        b, w, h, E = X.shape
        # H is the number of head(nHead = 8)    h is the height of image(32,224,...)
        if LOG.isPlot():
            show_tensors(X[0:64,:,:,0].contiguous().view(-1,1,w,h), nr_=8, pad_=2,title=f"X_E{LOG.epoch}_@{self.title}")
        if self.guided_filter is not None:
            X = self.guided_filter(X.permute(0,3,2,1)).permute(0,2,3,1)
            
        attention_probs = self.get_attention_probs(w, h)
        attention_probs = self.dropout(attention_probs)
        if False:
            input_values = torch.einsum('ijhkl,bkld->bijhd', attention_probs, hidden_states)
            input_values = input_values.contiguous().view(b, w, h, -1)
        else:   #just same as einsum 
            Ta = X.permute(0,3,1,2).reshape(-1,w*h)         #bEkl
            Tb = attention_probs.contiguous().permute(3,4,0,1,2).reshape(w*h,-1)         #klijH
            all_heads = (Ta@Tb).contiguous().reshape(b,E, w, h, -1).permute(0,2,3,1,4).reshape(b,w, h, -1)    #bEijH => bij(EH=1024)

        # if self.multiQKV is not None:
        #     all_heads += self.multiQKV(X)

        output_value = self.fc_allhead2hidden(all_heads)

        if self.multiQKV is not None:
            output_value += self.multiQKV(X)

        if LOG.isPlot():
            show_tensors(output_value[0:64,:,:,0].contiguous().view(-1,1,w,h), nr_=8, pad_=2,title=f"Result_E{LOG.epoch}_@{self.title}")

        if self.output_attentions:
            return attention_probs, output_value
        else:
            return output_value
   
def some_test():    
    import torchvision
    batch_tensor = torch.randn(*(10,1, 256, 256))   # (N, C, H, W)
    show_tensors(batch_tensor)
    grid_img = torchvision.utils.make_grid(batch_tensor, nrow=5, padding=10)
    print(grid_img.shape)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
    sys.exit(-1)

if __name__ == "__main__":
    from some_utils import *
    # some_test()
    import skimage.data
    astronaut = skimage.data.astronaut()
    #astronaut[...,0] = astronaut[...,0].T
    astronaut = numpy.moveaxis(astronaut,-1,0)[None,...]
    astronaut = torch.from_numpy(astronaut).float()
    isCuda = True
    gb = GaborFilters(in_channels=3)
    if isCuda:
        gb.cuda()
        astronaut = astronaut.cuda()

    res = gb(astronaut)
    print(f"result after filter={res.shape}")

    show_each_channel(res,picks=[0,1,2,8,9,-2,-1])
