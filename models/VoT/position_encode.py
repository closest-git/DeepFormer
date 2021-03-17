import math
import numpy as np
import torch
import torch.nn as nn
from enum import Enum

#   https://github.com/wzlxjtu/PositionalEncoding2D

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        _, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x,self.channels),device=tensor.device).type(tensor.type())
        emb[:,:self.channels] = emb_x

        return emb[None,:,:orig_ch]

class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)        
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0,2,1)
        enc = self.penc(tensor)
        return enc.permute(0,2,1)

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels/2))
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        _, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x,y,self.channels*2),device=tensor.device).type(tensor.type())
        emb[:,:,:self.channels] = emb_x
        emb[:,:,self.channels:2*self.channels] = emb_y

        # return emb[None,:,:,:orig_ch]
        return emb[None,:,:,:1]

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)        
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0,2,3,1)
        enc = self.penc(tensor)
        return enc.permute(0,3,1,2)


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels/3))
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        _, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x,y,z,self.channels*3),device=tensor.device).type(tensor.type())
        emb[:,:,:,:self.channels] = emb_x
        emb[:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,2*self.channels:] = emb_z

        return emb[None,:,:,:,:orig_ch]

class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)        
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0,2,3,4,1)
        enc = self.penc(tensor)
        return enc.permute(0,4,1,2,3)

class EmbeddingPaddingMode(Enum):
    Edge = 0
    Zero = 1
    Extend = 2  # only applicable with Fixed and Hybrid PositionEmbeddingTypes


class PositionEmbeddingType(Enum):
    Fixed = 0
    Learned = 1
    Hybrid = 2


class KeyStartPosition(Enum):
    """
    q1,q2,q3
    k1,k2,k3,k4,k5
    or
          q1,q2,q3
    k1,k2,k3,k4,k5
    """
    BeforeQuery = 0
    WithQuery = 1


# https://github.com/Separius/CudaRelativeAttention/blob/master/relative_embedding.py
class DistanceEmbedding(nn.Module):
    def __init__(self, depth, max_relative_position_past, max_relative_position_future, num_heads,
                 heads_share_relative_embedding, embedding_padding_mode, position_embedding_type, key_start_position):
        super().__init__()
        self.depth = depth
        self.max_relative_position_past = max_relative_position_past + 1  # count rel_dist==0 as past
        self.max_relative_position_future = max_relative_position_future
        self.heads_share_relative_embedding = heads_share_relative_embedding
        self.embedding_padding_mode = embedding_padding_mode
        self.position_embedding_type = position_embedding_type
        self.key_start_position = key_start_position
        if position_embedding_type == PositionEmbeddingType.Learned:
            assert embedding_padding_mode != EmbeddingPaddingMode.Extend
            if heads_share_relative_embedding:
                embedding_shape = (depth, max_relative_position_past + max_relative_position_future)
            else:
                embedding_shape = (num_heads, depth, max_relative_position_past + max_relative_position_future)
            self.embedding = nn.Parameter(torch.empty(embedding_shape))
            nn.init.normal_(self.embedding, mean=0, std=depth ** -0.5)
            self.last_past = max_relative_position_past
            self.last_future = max_relative_position_future
        else:
            self.register_buffer('_float_tensor', torch.FloatTensor(1))
            self.last_past = None
            self.last_future = None
            self.embedding = self.get_sinusoidal_embedding(self.max_relative_position_past,
                                                           self.max_relative_position_future)
            if position_embedding_type == PositionEmbeddingType.Fixed:
                assert heads_share_relative_embedding
            if position_embedding_type == PositionEmbeddingType.Hybrid:
                if heads_share_relative_embedding:
                    self.weight = nn.Parameter(torch.eye(depth))
                else:
                    self.weight = nn.Parameter(torch.eye(depth).unsqueeze(0).repeat(num_heads, 1, 1))

    def get_sinusoidal_embedding(self, past, future):
        if self.last_past is not None and past <= self.last_past and \
                self.last_future is not None and future <= self.last_future:
            emb = self.embedding.to(self._float_tensor)
        else:
            num_embeddings = past + future
            half_dim = self.depth // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
            emb = (torch.arange(num_embeddings, dtype=torch.float) - past + 1).unsqueeze(0) * emb.unsqueeze(1)
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=0).view(-1, num_embeddings)
            if self.depth % 2 == 1:
                emb = torch.cat([emb, torch.zeros(1, num_embeddings)], dim=0)
            emb = emb.to(self._float_tensor)
            self.last_past = past
            self.last_future = future
        self.embedding = emb
        return self.embedding

    @staticmethod
    def matmul_with_relative_keys(query, distance_embedding, heads_share_relative_embedding, bias=None):
        """Helper function for dot_product_unmasked_self_attention_relative_nd.
        Args:
            query: [batch, heads, None or T, None or H, W, d]
            distance_embedding: [None or heads, d, length]
            bias: Optional([heads, d])
        Returns:
            res: [batch, heads, None or T, None or H, W, length]
        """
        if bias is not None:
            # q is (B, N, ..., d) and bias is (N, d)
            query = query + bias.view(1, query.size(1), *([1] * (query.ndim - 3)), -1)
        dim_str = 'xyz'[:query.ndim - 3]
        head_str = '' if heads_share_relative_embedding else 'h'
        return torch.einsum(f'bh{dim_str}d,{head_str}dm->bh{dim_str}m', query, distance_embedding)

    def get_distance_embedding(self, q_len, k_len):
        if self.key_start_position == KeyStartPosition.BeforeQuery:
            assert q_len <= k_len
            past = k_len
            future = q_len - 1
        else:
            past = q_len
            future = k_len - 1
        if self.position_embedding_type == PositionEmbeddingType.Learned:
            initial_embedding = self.embedding  # (Nh or None, depth, max_past+max_future+1)
        elif self.embedding_padding_mode == EmbeddingPaddingMode.Extend:
            initial_embedding = self.get_sinusoidal_embedding(past, future)
        else:
            initial_embedding = self.embedding.to(self._float_tensor)
        initial_embedding = self.prune_embedding(past, future, initial_embedding)
        if self.position_embedding_type == PositionEmbeddingType.Hybrid:
            initial_embedding = torch.einsum('{h}ed, dt -> {h}et'.format(
                h='' if self.heads_share_relative_embedding else 'h'), self.weight, initial_embedding)
        if self.embedding_padding_mode == EmbeddingPaddingMode.Extend:
            return initial_embedding
        pad_shape = (max(past - self.last_past, 0), max(future - self.last_future, 0))
        if self.embedding_padding_mode == EmbeddingPaddingMode.Zero:
            return F.pad(initial_embedding, pad_shape, 'constant')
        if self.heads_share_relative_embedding:  # replicate padding does not work on 2d tensors
            return F.pad(initial_embedding.unsqueeze(0), pad_shape, 'replicate').squeeze(0)
        return F.pad(initial_embedding, pad_shape, 'replicate')

    def prune_embedding(self, past_len, future_len, embedding):
        return embedding[..., max(0, self.last_past - past_len):self.last_past + future_len]

    def forward(self, q_len, q=None, bias=None, k_len=None):
        if k_len is None:
            k_len = q_len
        distance_embedding = self.get_distance_embedding(q_len, k_len)
        if q is None:
            return distance_embedding
        return self.matmul_with_relative_keys(q, distance_embedding, self.heads_share_relative_embedding, bias)