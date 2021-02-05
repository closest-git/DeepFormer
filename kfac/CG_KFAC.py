import math
import torch
import time
import multiprocessing
from .kfac_preconditioner import *

class CG_KFAC(KFAC):
    def __init__(self,
                 model,
                 lr=0.1,
                 factor_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 fac_update_freq=10,
                 kfac_update_freq=100,
                 batch_averaged=True,
                 diag_blocks=1,
                 diag_warmup=0,
                 distribute_layer_factors=None):
        super(CG_KFAC,self).__init__(model,lr,factor_decay,damping,kl_clip,fac_update_freq,
            kfac_update_freq,batch_averaged,diag_blocks,diag_warmup,distribute_layer_factors)

        group = self.param_groups[0]
        self.lr = group['lr']
        self.nu = 1.0
        self.cur_accuracy=0
        self.last_accuracy=0
        self.damping = group['damping']
        self.fac_update_freq = group['fac_update_freq']
        self.kfac_update_freq = group['kfac_update_freq']

        self.last_x0 = {}
        for module in self.modules:
            self.last_x0[module] = None
        self.use_last_x0 = False
        self.adaptive_damping = True

        self.T_r = 0.01
        self.T_m = 10
        self.nIter = 0
        self.n_1 = 0
        self.n_2 = 0
        self.n_3 = 0
        self.T_all = 0

        self.mog_info = {}

    def __repr__(self):
        return f"lr={self.lr:.4f} damping={self.damping},adaptive_damping={self.adaptive_damping},fac_update_freq={self.fac_update_freq},\n \
            kfac_update_freq={self.kfac_update_freq} x0={self.use_last_x0},T_r={self.T_r},T_m={self.T_m}"

    
    def dump(self,nEpoch):
        info = f"lr={self.lr:.4f} nu={self.nu:.3f}  damping={self.damping:.1e},m={self.nIter*1.0/self.n_1:.2f} Bad={self.n_3}@{self.n_1} T={self.T_all/nEpoch:.1f}"
        return info
    
    def _init_A(self, factor, module):
        self.m_A[module] = torch.diag(factor.new(factor.shape[0]).fill_(1))

    def _init_G(self, factor, module):
        self.m_G[module] = torch.diag(factor.new(factor.shape[0]).fill_(1))
    
    def _clear_eigen(self):
        return
    
    def _update_scale_grad(self, updates):
        # vg_sum = 0
        # for module in self.modules:
        #     v = updates[module]
        #     vg_sum += (v[0] * module.weight.grad.data * self.lr ** 2).sum().item()
        #     if module.bias is not None:
        #         vg_sum += (v[1] * module.bias.grad.data * self.lr ** 2).sum().item()
        # nu = min(1.0, math.sqrt(self.kl_clip / abs(vg_sum)))

        for module in self.modules:
            v = updates[module]
            module.weight.grad.data.copy_(v[0])
            module.weight.grad.data.mul_(self.nu)
            if module.bias is not None:
                module.bias.grad.data.copy_(v[1])
                module.bias.grad.data.mul_(self.nu)

    
    def step(self, closure=None, epoch=None,accuracy=0):
        group = self.param_groups[0]
        self.lr = group['lr']
        # self.damping = group['damping']
        self.fac_update_freq = group['fac_update_freq']
        self.kfac_update_freq = group['kfac_update_freq']

        updates = {}
        handles = []

        if epoch is None:
            if self.diag_warmup > 0:
                print("WARNING: diag_warmup > 0 but epoch was not passed to "
                      "KFAC.step(). Defaulting to no diag_warmup")
            diag_blocks = self.diag_blocks
        else:
            diag_blocks = self.diag_blocks if epoch >= self.diag_warmup else 1

        if self.steps % self.fac_update_freq == 0:
            self._update_A()
            self._update_G()
            if hvd.size() > 1:
                self._allreduce_factors()
            

        # if we are switching from no diag approx to approx, we need to clear
        # off-block-diagonal elements
        if not self.have_cleared_Q and \
                epoch == self.diag_warmup and \
                self.steps % self.kfac_update_freq == 0:
            # self._clear_eigen()
            self.have_cleared_Q = True

        if self.steps % self.kfac_update_freq == 0:
            # reset rank iter so device get the same layers
            # to compute to take advantage of caching
            self.rank_iter.reset() 
            self.last_accuracy = self.cur_accuracy
            self.cur_accuracy = accuracy
            if self.cur_accuracy <= self.last_accuracy*1.001:
                self.nu = max(0.1,self.nu/1.01)
            else:
                self.nu = self.nu*1.01

            # for module in self.modules:
            #     # Get ranks to compute this layer on
            #     n = self._get_diag_blocks(module, diag_blocks)
            #     ranks_a = self.rank_iter.next(n)
            #     ranks_g = self.rank_iter.next(n) if self.distribute_layer_factors \
            #                                      else ranks_a

            #     self._update_eigen_A(module, ranks_a)
            #     self._update_eigen_G(module, ranks_g)

            # if hvd.size() > 1:
            #     self._allreduce_eigendecomp()
        self.isFail = False
        if False:        
            self.cys_grad(updates)
        else:
            for module in self.modules:     #To use CUDA with multiprocessing, you must use the 'spawn' start method
                # grad = self._get_grad(module)            
                self._get_preconditioned_grad(module,updates)            
                # updates[module] = precon_grad            

        self._update_scale_grad(updates)

        self.steps += 1
        #damping would be very SMALL(1.1e-06!!!)
        # if self.isFail:
        #     self.damping = self.damping*2
        # else:
        #     self.damping = self.damping/1.01        

    def OnLoss(self):
        isBad = False
        if isBad:
            self.nu = self.nu/1.01
        else:
            self.nu = self.nu*1.01

    def FV(self,a,g,v):          
        v1 = g.t()@ v @ a
        # v2 = g @ v1 @ a.t() + self.damping*v
        # return v2
        
        return v1+self.damping*v
    
    def CG_m(self,a,g,x0,b):
        # print(f"a={a.shape}\tg={g.shape}\tx0={x0.shape}")
        
        if x0 is None:
            r = b
            x = None
        else:
            r = b-self.FV(a,g,x0)
            x = x0
        p = r             
 
        rou_0 = torch.sum(r*r)            
        res_0 = rou_0
        m = 0  
        t0=time.time() 
        while m<self.T_m:            
            u = self.FV(a,g,p)
            s = torch.sum(p*u)
            alpha = rou_0/s
            x = alpha*p if x is None else x+alpha*p
            r = r-alpha*u
            rou_1 = torch.sum(r*r)
            if(rou_1/res_0<self.T_r):
                self.n_2=self.n_2;    break #return x,m+1

            beta = rou_1/rou_0
            p = r + beta*p
            rou_0 = rou_1
            m=m+1
        
            self.T_all += time.time()-t0   
        # print(f"res={res_0:.3f}->{rou_1.item():.6f}\talpha={alpha.item():.3f}\tbeta={beta.item():.3f}")
        if(rou_1>res_0 or rou_1<=0):    
            self.isFail = True        
            self.n_3=self.n_3+1;    return b,m+1
        return x,m+1

    def _get_preconditioned_grad(self, module,updates):           # 
        grad = self._get_grad(module)  
        self.n_1=self.n_1+1
        if self.use_last_x0 and self.last_x0[module] is None:
            self.last_x0[module] = torch.zeros_like(grad)     
            
        v,nIter = self.CG_m(self.m_A[module],self.m_G[module],self.last_x0[module],grad)
        
        if self.use_last_x0:
            self.last_x0[module] = v
        self.nIter=self.nIter+nIter
        if module.bias is not None:
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(module.weight.grad.data.size()) # weight
            v[1] = v[1].view(module.bias.grad.data.size())   # bias
        else:
            v = [v.view(module.weight.grad.data.size())]
        updates[module] = v
        return   

    def FV_all(self,v000):    
        for module in self.modules:
            g=self.m_G[module]
            a=self.m_A[module]
            info = self.mog_info[module]
            v = v000[info[1]:info[2]].view(info[0])      
            v1 = g.t()@ v @ a+self.damping*v
            self.cg_u[info[1]:info[2]] = v1.view(-1)
            assert(torch.numel(v1)==info[2]-info[1])

        return self.cg_u

    def CG_all(self,b):
        # print(f"a={a.shape}\tg={g.shape}\tx0={x0.shape}")
        r = b
        x = None
        p = r             
 
        rou_0 = torch.sum(r*r)            
        res_0 = rou_0
        m = 0  
        t0=time.time() 
        while m<self.T_m:            
            u = self.FV_all(p)
            s = torch.sum(p*u)
            alpha = rou_0/s
            x = alpha*p if x is None else x+alpha*p
            r = r-alpha*u
            rou_1 = torch.sum(r*r)
            if(rou_1/res_0<self.T_r):
                self.n_2=self.n_2;    break #return x,m+1

            beta = rou_1/rou_0
            p = r + beta*p
            rou_0 = rou_1
            m=m+1
        
        self.T_all += time.time()-t0   
        # print(f"res={res_0:.3f}->{rou_1.item():.6f}\talpha={alpha.item():.3f}\tbeta={beta.item():.3f}")
        if(rou_1>res_0 or rou_1<=0):            
            self.n_3=self.n_3+1;    return b,m+1
        return x,m+1

    def cys_grad(self,updates):
        self.n_1=self.n_1+1    
        grad_ = []
        nz = 0
        for module in self.modules:     #To use CUDA with multiprocessing, you must use the 'spawn' start method
            g_ = self._get_grad(module)
            n = torch.numel(g_)
            self.mog_info[module]=[g_.shape,nz,nz+n]
            grad_.append(g_.view(-1))
            nz = nz+n
        grad = torch.cat(grad_, dim=0)
        self.cg_u = torch.zeros_like(grad)
        
        
        x,nIter = self.CG_all(grad)   
        self.nIter=self.nIter+nIter
        
        k0,k1=0,0
        for module in self.modules:
            info = self.mog_info[module]
            v = x[info[1]:info[2]].view(info[0])
            if module.bias is not None:
                v = [v[:, :-1], v[:, -1:]]
                v[0] = v[0].view(module.weight.grad.data.size()) # weight
                v[1] = v[1].view(module.bias.grad.data.size())   # bias
            else:
                v = [v.view(module.weight.grad.data.size())]
            updates[module] = v

         
