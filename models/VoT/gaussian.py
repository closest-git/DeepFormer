import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from .bert_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME
from vit_pytorch import sparsemax, entmax15
from .guided_filter import SelfGuidedFilter
import numbers


def gaussian_kernel_2d(mean, std_inv, size):
    """Create a 2D gaussian kernel

    Args:
        mean: center of the gaussian filter (shift from origin)
            (2, ) vector
        std_inv: standard deviation $Sigma^{-1/2}$
            can be a single number, a vector of dimension 2, or a 2x2 matrix
        size: size of the kernel
            pair of integer for width and height
            or single number will be used for both width and height

    Returns:
        A gaussian kernel of shape size.
    """
    if type(mean) is torch.Tensor:
        device = mean.device
    elif type(std_inv) is torch.Tensor:
        device = std_inv.device
    else:
        device = "cpu"

    # repeat the size for width, height if single number
    if isinstance(size, numbers.Number):
        width = height = size
    else:
        width, height = size

    # expand std to (2, 2) matrix
    if isinstance(std_inv, numbers.Number):
        std_inv = torch.tensor([[std_inv, 0], [0, std_inv]], device=device)
    elif std_inv.dim() == 0:
        std_inv = torch.diag(std_inv.repeat(2))
    elif std_inv.dim() == 1:
        assert len(std_inv) == 2
        std_inv = torch.diag(std_inv)

    # Enforce PSD of covariance matrix
    covariance_inv = std_inv.transpose(0, 1) @ std_inv
    covariance_inv = covariance_inv.float()

    # make a grid (width, height, 2)
    X = torch.cat(
        [
            t.unsqueeze(-1)
            for t in reversed(
                torch.meshgrid(
                    [torch.arange(s, device=device) for s in [width, height]]
                )
            )
        ],
        dim=-1,
    )
    X = X.float()

    # center the gaussian in (0, 0) and then shift to mean
    X -= torch.tensor([(width - 1) / 2, (height - 1) / 2], device=device).float()
    X -= mean.float()

    # does not use the normalize constant of gaussian distribution
    Y = torch.exp((-1 / 2) * torch.einsum("xyi,ij,xyj->xy", [X, covariance_inv, X]))

    # normalize
    # TODO could compute the correct normalization (1/2pi det ...)
    # and send warning if there is a significant diff
    # -> part of the gaussian is outside the kernel
    Z = Y / Y.sum()
    return Z


class se_reponse(nn.Module):
    def __init__(self, nTree, reduction=16):
        super(se_reponse, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.nTree = nTree
        self.reduction = reduction
        self.nEmbed = max(2,self.nTree//reduction)
        
        self.fc = nn.Sequential(
            nn.Linear(nTree, self.nEmbed, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.nEmbed, nTree, bias=False),
            #nn.Sigmoid()
            nn.Softmax()
        )
    
    def forward(self, x):
        b, t, _ = x.size()
        y = torch.mean(x,dim=-1)
        y = self.fc(y)
        out = torch.einsum('btr,bt->btr', x,y) 
        # dist = torch.dist(out,out_0,2)
        # assert dist==0
        return out

class eca_input(nn.Module):
    def __init__(self, nFeat, k_size=3):
        super(eca_input, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.nFeat = nFeat
        self.k_size = k_size
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
    
    def forward(self, x):
        b, f = x.size()
        #y = torch.mean(x,dim=0)
        y = x
        y = self.conv(y.unsqueeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1)
        y = F.sigmoid(y)
        #y = F.softmax(y)
        out = torch.einsum('bf,bf->bf', x,y) 
        return out

class se_basic(nn.Module):
    def __init__(self, nFeat, reduction=4):
        super(se_basic, self).__init__()
        self.nFeat = nFeat
        self.reduction = 1
        self.nEmbed = self.nFeat//reduction #max(2,self.nFeat//reduction)
        
        self.fc = nn.Sequential(
            nn.Linear(self.nFeat, self.nEmbed, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.nEmbed, self.nFeat, bias=False),
            #nn.Sigmoid()
            nn.Softmax()
        )
    
    def forward(self, x):
        b, f = x.size()
        y = torch.mean(x,dim=0)
        y = self.fc(y)
        out = torch.einsum('bf,f->b', x,y) 
        out = out.squeeze()
        return out

class GaussianSelfAttention(nn.Module):
    def __init__(self, config, hidden_in,output_attentions=False, keep_multihead_output=False):
        super().__init__()
        self.attention_gaussian_blur_trick = config.attention_gaussian_blur_trick
        self.attention_isotropic_gaussian = config.attention_isotropic_gaussian
        self.gaussian_init_mu_std = config.gaussian_init_mu_std
        self.gaussian_init_sigma_std = config.gaussian_init_sigma_std
        self.config = config
        # self.attention_dropout = nn.Dropout(p=0.1)
        self.guided_filter = None   #SelfGuidedFilter(3,8,8)

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = hidden_in    #config.hidden_size
        # assert config.hidden_size % config.num_attention_heads == 0, "num_attention_heads should divide hidden_size"
        # self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * hidden_in   #config.hidden_size
        self.output_attentions = output_attentions
        self.se_excite = None   #se_basic(self.num_attention_heads,reduction=1)

        # CAREFUL: if change something here, change also in reset_heads (TODO remove code duplication)
        # shift of the each gaussian per head
        self.attention_centers = nn.Parameter(
            torch.zeros(self.num_attention_heads, 2).normal_(0.0, config.gaussian_init_mu_std)
        )
        self.isSigma = True   

        if config.attention_isotropic_gaussian:
            # only one scalar (inverse standard deviation)
            # initialized to 1 + noise
            attention_spreads = 1 + torch.zeros(self.num_attention_heads).normal_(0, config.gaussian_init_sigma_std)
        else:
            # Inverse standart deviation $Sigma^{-1/2}$
            # 2x2 matrix or a scalar per head
            # initialized to noisy identity matrix
            attention_spreads = torch.eye(2).unsqueeze(0).repeat(self.num_attention_heads, 1, 1)
            attention_spreads += torch.zeros_like(attention_spreads).normal_(0, config.gaussian_init_sigma_std)

        if self.isSigma:
            self.attention_spreads = attention_spreads
            attention_spreads = self.get_heads_target_vectors()
            print(attention_spreads)
            self.sigmaLayer = nn.Linear(5,self.num_attention_heads)   
            with torch.no_grad():
                self.sigmaLayer.weight.copy_(attention_spreads)
        else:
            self.attention_spreads = nn.Parameter(attention_spreads)

        self.isMaxout = False   #useless!!!
        if self.isMaxout:
            pass
        else:
            self.fc_allhead2hidden = nn.Linear(self.all_head_size,hidden_in )      #config.hidden_size

        if not config.attention_gaussian_blur_trick:
            # relative encoding grid (delta_x, delta_y, delta_x**2, delta_y**2, delta_x * delta_y)
            MAX_WIDTH_HEIGHT = 50
            range_ = torch.arange(MAX_WIDTH_HEIGHT)
            grid = torch.cat([t.unsqueeze(-1) for t in torch.meshgrid([range_, range_])], dim=-1)
            relative_indices = grid.unsqueeze(0).unsqueeze(0) - grid.unsqueeze(-2).unsqueeze(-2)        #so elegant and hard code
            R = torch.cat([relative_indices, relative_indices ** 2, (relative_indices[..., 0] * relative_indices[..., 1]).unsqueeze(-1)], dim=-1)
            R = R.float()
            self.register_buffer("R", R)
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def get_heads_target_vectors(self):
        if self.attention_isotropic_gaussian:
            a = c = self.attention_spreads ** 2
            b = torch.zeros_like(self.attention_spreads)
        else:
            # $\Sigma^{-1}$
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
            # u = self.attention_spreads
            d = 5
            R1 = self.R[:width,:height,:width,:height,:].reshape(-1,d).contiguous()
            attention_scores = self.sigmaLayer(R1)  #R1@(u.permute(1,0))
            attention_scores = attention_scores.view(width,height,width,height,-1).permute(0,1,4,2,3).contiguous()
        else:
            u = self.get_heads_target_vectors()
        # Compute attention map for each head
            attention_scores = torch.einsum('ijkld,hd->ijhkl', [self.R[:width,:height,:width,:height,:], u])
        # Softmax
        # attention_scores = self.attention_dropout(attention_scores)
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores.view(width, height, self.num_attention_heads, -1))
        # attention_probs = entmax15(attention_scores.view(width, height, self.num_attention_heads, -1),dim=-1)
        attention_probs = attention_probs.view(width, height, self.num_attention_heads, width, height)

        return attention_probs

    def reset_heads(self, heads):
        device = self.attention_spreads.data.device
        reset_heads_mask = torch.zeros(self.num_attention_heads, device=device, dtype=torch.bool)
        for head in heads:
            reset_heads_mask[head] = 1

        # Reinitialize mu and sigma of these heads
        self.attention_centers.data[reset_heads_mask].zero_().normal_(0.0, self.gaussian_init_mu_std)

        if self.attention_isotropic_gaussian:
            self.attention_spreads.ones_().normal_(0, self.gaussian_init_sigma_std)
        else:
            self.attention_spreads.zero_().normal_(0, self.gaussian_init_sigma_std)
            self.attention_spreads[:, 0, 0] += 1
            self.attention_spreads[:, 1, 1] += 1

        # Reinitialize value matrix for these heads
        mask = torch.zeros(self.num_attention_heads, self.attention_head_size, dtype=torch.bool)
        for head in heads:
            mask[head] = 1
        mask = mask.view(-1).contiguous()
        self.value.weight.data[:, mask].normal_(mean=0.0, std=self.config.initializer_range)
        # self.value.bias.data.zero_()


    def blured_attention(self, X):
        """Compute the weighted average according to gaussian attention without
        computing explicitly the attention coefficients.

        Args:
            X (tensor): shape (batch, width, height, dim)
        Output:
            shape (batch, width, height, dim x num_heads)
        """
        num_heads = self.attention_centers.shape[0]
        batch, width, height, d_total = X.shape
        Y = X.permute(0, 3, 1, 2).contiguous()

        kernels = []
        kernel_width = kernel_height = 7
        assert kernel_width % 2 == 1 and kernel_height % 2 == 1, 'kernel size should be odd'

        for mean, std_inv in zip(self.attention_centers, self.attention_spreads):
            conv_weights = gaussian_kernel_2d(mean, std_inv, size=(kernel_width, kernel_height))
            conv_weights = conv_weights.view(1, 1, kernel_width, kernel_height).repeat(d_total, 1, 1, 1)
            kernels.append(conv_weights)

        weights = torch.cat(kernels)

        padding_width = (kernel_width - 1) // 2
        padding_height = (kernel_height - 1) // 2
        out = F.conv2d(Y, weights, groups=d_total, padding=(padding_width, padding_height))

        # renormalize for padding
        all_one_input = torch.ones(1, d_total, width, height, device=X.device)
        normalizer = F.conv2d(all_one_input, weights,  groups=d_total, padding=(padding_width, padding_height))
        out /= normalizer

        return out.permute(0, 2, 3, 1).contiguous()

    def forward(self, hidden_states, attention_mask, head_mask=None):
        assert len(hidden_states.shape) == 4
        b, w, h, E = hidden_states.shape
        # H is the number of head(nHead = 8)    h is the height of image(32,224,...)
        if self.guided_filter is not None:
            hidden_states = self.guided_filter(hidden_states.permute(0,3,2,1)).permute(0,2,3,1)
            
        if not self.attention_gaussian_blur_trick:
            attention_probs = self.get_attention_probs(w, h)
            attention_probs = self.dropout(attention_probs)
            if False:
                input_values = torch.einsum('ijhkl,bkld->bijhd', attention_probs, hidden_states)
                input_values = input_values.contiguous().view(b, w, h, -1)
            else:   #just same as einsum 
                Ta = hidden_states.permute(0,3,1,2).reshape(-1,w*h)         #bEkl
                Tb = attention_probs.contiguous().permute(3,4,0,1,2).reshape(w*h,-1)         #klijH
                all_heads = (Ta@Tb).contiguous().reshape(b,E, w, h, -1).permute(0,2,3,1,4).reshape(b,w, h, -1)    #bEijH => bij(EH=1024)
        else:
            all_heads = self.blured_attention(hidden_states)        

        if self.isMaxout:
            all_heads = all_heads.reshape(-1, self.num_attention_heads)
            # h_norm = torch.norm(all_heads, dim=0)
            # _,max_id = torch.max(h_norm,0)
            # max_id = 0
            # output_value = all_heads[:,max_id].reshape(b,w, h, -1)
            output_value =  torch.mean(all_heads, dim=1).reshape(b,w, h, -1)            
        elif self.se_excite is not None:
            all_heads = all_heads.reshape(-1, self.num_attention_heads)
            output_value = self.se_excite(all_heads).reshape(b,w, h, -1)   
        else:            
            output_value = self.fc_allhead2hidden(all_heads)

        if self.output_attentions:
            return attention_probs, output_value
        else:
            return output_value

