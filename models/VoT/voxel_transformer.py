# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
import numbers
import torchvision

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from .bert_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME
from .gaussian import *
from .gabor_filter import *
# from vit_pytorch import sparsemax, entmax15
# from .guided_filter import SelfGuidedFilter

MAX_WIDTH_HEIGHT = 64

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
    "bert-base-german-cased": "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased.tar.gz",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking.tar.gz",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking.tar.gz",
}
BERT_CONFIG_NAME = "bert_config.json"
TF_WEIGHTS_NAME = "model.ckpt"



def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    device = layer.weight.device
    index = index.to(device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer.to(device)


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                l = re.split(r"_(\d+)", m_name)
            else:
                l = [m_name]
            if l[0] == "kernel" or l[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif l[0] == "output_bias" or l[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif l[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif l[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(
        self,
        vocab_size_or_config_json_file,
        hidden_size=None,
        position_encoding_size=None,
        num_hidden_layers=None,
        num_attention_heads=None,
        intermediate_size=None,
        hidden_act=None,
        hidden_dropout_prob=None,
        attention_probs_dropout_prob=None,
        max_position_embeddings=None,
        type_vocab_size=None,
        initializer_range=None,
        layer_norm_eps=None,
        use_learned_2d_encoding=None,
        share_position_encoding=None,
        use_attention_data=None,
        query_positional_score=None,
        # use_gaussian_attention=None,
        use_attention="gaussian",
        add_positional_encoding_to_input=None,
        positional_encoding=None,
        max_positional_encoding=None,
        attention_gaussian_blur_trick=None,
        attention_isotropic_gaussian=None,
        gaussian_init_sigma_std=None,
        gaussian_init_mu_std=None,
    ):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (
            sys.version_info[0] == 2 and isinstance(vocab_size_or_config_json_file, unicode)
        ):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.position_encoding_size = position_encoding_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.use_learned_2d_encoding = use_learned_2d_encoding
            self.use_attention = use_attention
            self.positional_encoding = positional_encoding
            self.max_positional_encoding = max_positional_encoding
            self.attention_gaussian_blur_trick = attention_gaussian_blur_trick
            self.attention_isotropic_gaussian = attention_isotropic_gaussian
            self.gaussian_init_sigma_std = gaussian_init_sigma_std
            self.gaussian_init_mu_std = gaussian_init_mu_std
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as Former_LayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex ."
    )
    # from torch.nn import LayerNorm as Former_LayerNorm
    from torch.nn import Identity as Former_LayerNorm
    # from torch.nn import BatchNorm2d as Former_LayerNorm

    # class BertLayerNorm(nn.Module):
    #     def __init__(self, hidden_size, eps=1e-12):
    #         """Construct a layernorm module in the TF style (epsilon inside the square root).
    #         """
    #         super(BertLayerNorm, self).__init__()
    #         # self.weight = nn.Parameter(torch.ones(hidden_size))
    #         # self.bias = nn.Parameter(torch.zeros(hidden_size))
    #         self.variance_epsilon = eps

    #     def forward(self, x):
    #         u = x.mean(-1, keepdim=True)
    #         s = (x - u).pow(2).mean(-1, keepdim=True)
    #         x = (x - u) / torch.sqrt(s + self.variance_epsilon)
    #         return x
    #         # return self.weight * x + self.bias





class Learned2DRelativeSelfAttention(nn.Module):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super().__init__()
        self.output_attentions = output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.use_attention_data = config.use_attention_data
        self.query_positional_score = config.query_positional_score
        self.hidden_size = config.hidden_size
        self.all_head_size = config.hidden_size * self.num_attention_heads

        max_position_embeddings = config.max_position_embeddings

        position_embedding_size = config.hidden_size
        if self.query_positional_score:
            position_embedding_size = config.hidden_size // 2
        if config.position_encoding_size != -1:
            position_embedding_size = config.position_encoding_size

        self.row_embeddings = nn.Embedding(2 * max_position_embeddings - 1, position_embedding_size)
        self.col_embeddings = nn.Embedding(2 * max_position_embeddings - 1, position_embedding_size)

        if not self.query_positional_score:
            self.head_keys_row = nn.Linear(position_embedding_size, self.num_attention_heads, bias=False)
            self.head_keys_col = nn.Linear(position_embedding_size, self.num_attention_heads, bias=False)

        # need query linear transformation
        if self.use_attention_data or self.query_positional_score:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)

        # need key linear transformation
        if self.use_attention_data:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.value = nn.Linear(self.all_head_size, config.hidden_size)

        deltas = torch.arange(max_position_embeddings).view(1, -1) - torch.arange(max_position_embeddings).view(-1, 1)
        # shift the delta to [0, 2 * max_position_embeddings - 1]
        relative_indices = deltas + max_position_embeddings - 1

        self.register_buffer("relative_indices", relative_indices)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        assert len(hidden_states.shape) == 4
        b, w, h, c = hidden_states.shape

        # -- B, W, H, num_heads, W, H
        attention_scores, attention_scores_per_type = self.compute_attention_scores(hidden_states)
        shape = attention_scores.shape
        attention_probs = nn.Softmax(dim=-1)(attention_scores.view(*shape[:-2], -1)).view(shape)
        # expand batch dim if 1
        if shape[0] != b:
            attention_probs = attention_probs.expand(b, *shape[1:])

        attention_probs = self.dropout(attention_probs)

        input_values = torch.einsum('bijhkl,bkld->bijhd', attention_probs, hidden_states)
        input_values = input_values.contiguous().view(b, w, h, -1)
        output_value = self.value(input_values)

        if self.output_attentions:
            attention_scores_per_type["attention_scores"] = attention_scores
            attention_scores_per_type["attention_probs"] = attention_probs
            return attention_scores_per_type, output_value
        else:
            return output_value

    def compute_attention_scores(self, hidden_states):
        """Compute the positional attention for an image of size width x height
        Returns: tensor of attention scores (1 or batch, width, height, num_head, width, height)

        Attention scores:
            * Position only
                Options: use_attention_data=False, query_positional_score=False
                w_q^T * r
                where w_q is a learned vector per head
            * Query and positional encoding (without query key attention scores),
                same as q * r in (Ramachandran et al., 2019)
                Options: use_attention_data=False, query_positional_score=True
                X * W_Q * r
            * With data
                same as q*k + q*r in (Ramachandran et al., 2019)
                Options: use_attention_data=True, query_positional_score=True
                X * W_Q * W_K^T * X^T + X * W_Q * r
            * Last option use_attention_data=True, query_positional_score=False was not used
        """
        batch_size, height, width, hidden_dim = hidden_states.shape

        # compute query data if needed
        if self.use_attention_data or self.query_positional_score:
            q = self.query(hidden_states)
            q = q.view(batch_size, width, height, self.num_attention_heads, self.hidden_size)

        # compute key data if needed
        if self.use_attention_data:
            k = self.key(hidden_states)
            k = k.view(batch_size, width, height, self.num_attention_heads, self.hidden_size)

        # Compute attention scores based on position
        # Probably not optimal way to order computation
        relative_indices = self.relative_indices[:width,:width].reshape(-1)
        row_embeddings = self.row_embeddings(relative_indices)

        relative_indices = self.relative_indices[:height,:height].reshape(-1)
        col_embeddings = self.col_embeddings(relative_indices)

        # keep attention scores/prob for plotting
        attention_scores_per_type = {}
        sqrt_normalizer = math.sqrt(self.hidden_size)

        if not self.query_positional_score:
            # Caveat: sqrt rescaling is not used in this case
            row_scores = self.head_keys_row(row_embeddings).view(1, width, 1, width, self.num_attention_heads)
            col_scores = self.head_keys_col(col_embeddings).view(height, 1, height, 1, self.num_attention_heads)
            # -- H, W, H, W, num_attention_heads
            attention_scores = row_scores + col_scores
            # -- H, W, num_attention_heads, H, W
            attention_scores = attention_scores.permute(0, 1, 4, 2, 3)
            # -- 1, H, W, num_attention_heads, H, W
            attention_scores = attention_scores.unsqueeze(0)

            attention_scores_per_type["w_q^Tr"] = attention_scores

        else:  # query_positional_score
            # B, W, H, num_attention_heads, D // 2
            q_row = q[:, :, :, :, :self.hidden_size // 2]
            q_col = q[:, :, :, :, self.hidden_size // 2:]

            row_scores = torch.einsum("bijhd,ikd->bijhk", q_row, row_embeddings.view(width, width, -1))
            col_scores = torch.einsum("bijhd,jld->bijhl", q_col, col_embeddings.view(height, height, -1))

            # -- B, H, W, num_attention_heads, H, W
            attention_scores = row_scores.unsqueeze(-1) + col_scores.unsqueeze(-2)
            attention_scores = attention_scores / sqrt_normalizer

            # save
            attention_scores_per_type["q^Tr"] = attention_scores

        # Compute attention scores based on data
        if self.use_attention_data:
            attention_content_scores = torch.einsum("bijhd,bklhd->bijhkl", q, k)
            attention_content_scores = attention_content_scores / sqrt_normalizer
            attention_scores = attention_scores + attention_content_scores
            
            # save
            attention_scores_per_type["q^Tk"] = attention_content_scores

        return attention_scores, attention_scores_per_type

    def get_attention_probs(self, width, height):
        """LEGACY
        Compute the positional attention for an image of size width x height
        Returns: tensor of attention probabilities (width, height, num_head, width, height)
        """
        relative_indices = self.relative_indices[:width,:width].reshape(-1)
        row_scores = self.head_keys_row(self.row_embeddings(relative_indices)).view(1, width, 1, width, self.num_attention_heads)

        relative_indices = self.relative_indices[:height,:height].reshape(-1)
        col_scores = self.head_keys_col(self.col_embeddings(relative_indices)).view(height, 1, height, 1, self.num_attention_heads)

        # -- H, W, H, W, num_attention_heads
        attention_scores = row_scores + col_scores
        # -- H, W, num_attention_heads, H, W
        attention_scores = attention_scores.permute(0, 1, 4, 2, 3)

        # -- H, W, num_attention_heads, H, W
        flatten_shape = [height, width, self.num_attention_heads, height * width]
        unflatten_shape = [height, width, self.num_attention_heads, height, width]
        attention_probs = nn.Softmax(dim=-1)(attention_scores.view(*flatten_shape)).view(*unflatten_shape)

        return attention_probs





# class SublayerConnection(nn.Module):
#     """
#     A residual connection followed by a layer norm.
#     Note for code simplicity the norm is first as opposed to last.
#     """

#     def __init__(self, size, dropout):
#         super(SublayerConnection, self).__init__()
#         self.norm = LayerNorm(size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, sublayer):
#         "Apply residual connection to any sublayer with the same size."
#         return x + self.dropout(sublayer(self.norm(x)))
class Residual_Noraml(nn.Module):
    def __init__(self,hidden_in, config):
        super(Residual_Noraml, self).__init__()
        self.dense = nn.Linear(hidden_in,hidden_in) #config.hidden_size, config.hidden_size
        self.LayerNorm = Former_LayerNorm(hidden_in, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VoxAttention(nn.Module):
    def __init__(self, config,hidden_in, output_attentions=False, keep_multihead_output=False,title=""):
        super(VoxAttention, self).__init__()
        self.output_attentions = output_attentions
        self.use_attention = config.use_attention
        self.title = title
        #   MHSA - multi-head self-attention
        if False:   #v0.1
            self.flatten_image = not config.use_gaussian_attention and not config.use_learned_2d_encoding
            self.use_gaussian_attention = config.use_gaussian_attention           

            assert not config.use_gaussian_attention or not config.use_learned_2d_encoding  # TODO change to enum args

            if config.use_gaussian_attention:
                attention_cls = GaussianSelfAttention
            elif config.use_learned_2d_encoding:
                attention_cls = Learned2DRelativeSelfAttention
            else:
                attention_cls = BertSelfAttention
            self.MHSA = attention_cls(config,hidden_in, output_attentions=output_attentions, keep_multihead_output=keep_multihead_output)
        else:   #more attention
            self.flatten_image = self.use_attention=="v0"            
            if self.use_attention=="gaussian":
                attention_cls = GaussianSelfAttention
            elif self.use_attention=="learned_2d_encoding":
                attention_cls = Learned2DRelativeSelfAttention
            elif self.use_attention=="gabor":
                attention_cls = GaborSelfAttention
            else:
                attention_cls = BertSelfAttention
            self.MHSA = attention_cls(config,hidden_in, output_attentions=output_attentions, keep_multihead_output=keep_multihead_output,title=self.title)
        self.config = config

        # self.self = attention_cls(config,hidden_in, output_attentions=output_attentions, keep_multihead_output=keep_multihead_output)
        self.residual = None    #Residual_Noraml(hidden_in,config)

    def prune_heads(self, heads):
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        dim = 1
        # Prune linear layers
        if not self.use_attention=="gaussian":         #self.use_gaussian_attention:
            self.self.query = prune_linear_layer(self.self.query, index)
            self.self.key = prune_linear_layer(self.self.key, index)
            self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
            dim = 0 

        if self.use_attention=="gaussian":              #self.use_gaussian_attention:
            device = self.self.attention_spreads.data.device
            keep_heads = torch.ones(self.self.num_attention_heads, device=device, dtype=torch.bool)
            for head in heads:
                keep_heads[head] = 0
            self.self.attention_spreads.data = self.self.attention_spreads.data[keep_heads].contiguous()
            self.self.attention_centers.data = self.self.attention_centers.data[keep_heads].contiguous()

        # dim = 0 if not self.use_gaussian_attention else 1
        self.self.value = prune_linear_layer(self.self.value, index, dim=dim)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def reset_heads(self, heads):
        """Only for Gaussian Attention"""
        assert self.use_attention=="gaussian"       #self.use_gaussian_attention
        self.self.reset_heads(heads)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        LOG = self.config.logger
        is_image = len(input_tensor.shape) == 4
        if is_image and self.flatten_image:
            batch, width, height, d = input_tensor.shape
            input_tensor = input_tensor.view([batch, -1, d])

        mhsa_output = self.MHSA(input_tensor, attention_mask, head_mask)
        if self.output_attentions:
            attentions, mhsa_output = mhsa_output
        if self.residual is None:
            attention_output = mhsa_output
        else:
            attention_output = self.residual(mhsa_output, input_tensor)
        
        if LOG.isPlot():
            b, w, h, E = attention_output.shape
            show_tensors(attention_output[0:64,:,:,0].contiguous().view(-1,1,w,h), nr_=8, pad_=2,title=f"an1_E{LOG.epoch}_@{self.title}")

        if is_image and self.flatten_image:
            attention_output = attention_output.view([batch, width, height, -1])

        if self.output_attentions:
            return attentions, attention_output
        return attention_output


class BertOutput(nn.Module):
    def __init__(self,hidden_out, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, hidden_out)
        self.LayerNorm = Former_LayerNorm(hidden_out, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
class BertIntermediate(nn.Module):
    def __init__(self,hidden_in, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_in, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

#   https://ai.stackexchange.com/questions/15524/why-would-you-implement-the-position-wise-feed-forward-network-of-the-transforme
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, nIn,nHidden, config):
        super(PositionwiseFeedForward, self).__init__()
        # nX = config.intermediate_size
        nX = nHidden
        self.w_1 = nn.Linear(nIn, nX)
        self.w_2 = nn.Linear(nX, nIn)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = gelu
        # self.activation = nn.ReLU()     # maybe use ReLU

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
        # return self.dropout(self.activation(x))

class voxel_drift(nn.Module):
    "drift model by Yingshi Chen @2021/3/18"

    def __init__(self, nIn,nHidden, config):
        super(voxel_drift, self).__init__()
        assert nHidden>nIn*8
        nX = nHidden
        self.w_1 = nn.Linear(nIn, nX)
        self.w_2 = nn.Linear(nX, nIn)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = gelu
        # self.activation = nn.ReLU()     # maybe use ReLU

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class Encoder(nn.Module):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False,hidden_in=128,hidden_out=128,id=0):
        super(Encoder, self).__init__()
        self.name = f"Lay{id}"
        self.config = config
        self.output_attentions = output_attentions
        self.FFN_prehalf = PositionwiseFeedForward(hidden_in,config.intermediate_size,config)
        self.attention = VoxAttention(
            config,hidden_in, output_attentions=output_attentions, keep_multihead_output=keep_multihead_output,title=self.name
        )
        self.FFN = PositionwiseFeedForward(hidden_in,48,config)
        if self.FFN is None:
            self.intermediate = BertIntermediate(hidden_in,config)
            self.output = BertOutput(hidden_out,config)
        # self.log_writer = self.config.logger        

    def forward(self, hidden_states, attention_mask, head_mask=None):
        if self.FFN_prehalf is not None:        # Strang-Marchuk splitting scheme of convection-diffusion equation
            hidden_states = hidden_states+self.FFN_prehalf(hidden_states)
        attention_output = self.attention(hidden_states, attention_mask, head_mask)    
        nMostPic = 64    
        width,height = self.config.INPUT_W,  self.config.INPUT_H


        if self.output_attentions:
            attentions, attention_output = attention_output
            # pics = attention_output[0:63,:,:,0].contiguous().view(-1,1,8,8)
            pics = attentions.contiguous().view(-1,1,width,height)[0:nMostPic,:,:,:]
            LOG = self.config.logger
            if LOG is not None and LOG.batch_idx==0:
                grid = torchvision.utils.make_grid(pics,nrow=8,padding=2)      #torchvision.utils.make_grid(tensors,nrow=nr_, padding=pad_)
                title = f"{self.name}/{LOG.epoch}_{LOG.batch_idx}"
                # LOG.add_image(title, grid, LOG.epoch)
                # show_tensors(pics, nr_=8, pad_=10,title=f"{self.name}")
        if self.FFN is None:
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
        else:
            #self.ff = Residual(PreNorm(hidden, PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)))
            layer_output = attention_output+self.FFN(attention_output)

        if self.output_attentions:
            return attentions, layer_output
        return layer_output

       
class VoxTransformer(nn.Module):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False,hidden_dim=None):
        super(VoxTransformer, self).__init__()
        self.output_attentions = output_attentions
        # layer_constructor = lambda: Encoder(
        #     config, output_attentions=output_attentions, keep_multihead_output=keep_multihead_output
        # )
        # self.layer = nn.ModuleList([layer_constructor() for _ in range(config.num_hidden_layers)])
        self.layer = nn.ModuleList([Encoder(
             config, output_attentions=output_attentions, keep_multihead_output=keep_multihead_output,hidden_in=hidden_dim[id],hidden_out=hidden_dim[id+1],id=id
        ) for id in range(config.num_hidden_layers)])

        if config.use_learned_2d_encoding and config.share_position_encoding:
            for layer in self.layer[1:]:
                self.layer[0].attention.self.row_embeddings = layer.attention.self.row_embeddings
                self.layer[0].attention.self.col_embeddings = layer.attention.self.col_embeddings


    def forward(
        self, hidden_states, attention_mask, output_all_encoded_layers=True, head_mask=None
    ):
        all_encoder_layers = []
        all_attentions = []
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states, attention_mask, head_mask[i] if head_mask is not None else None
            )
            if self.output_attentions:
                attentions, hidden_states = hidden_states
                all_attentions.append(attentions)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        if self.output_attentions:
            return all_attentions, all_encoder_layers
        return all_encoder_layers









