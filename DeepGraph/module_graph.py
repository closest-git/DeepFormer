import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from collections import OrderedDict
from .stat_tree import *
from .reporter import *
# from torchstat import ModelHook
# from torchstat import StatTree, StatNode, report_format
# from torchstat import compute_madd
# from torchstat import compute_flops
# from torchstat import compute_memory

def compute_madd(module, inp, out):
    # if isinstance(module, nn.Conv2d):
    #     return compute_Conv2d_madd(module, inp, out)
    # elif isinstance(module, nn.ConvTranspose2d):
    #     return compute_ConvTranspose2d_madd(module, inp, out)
    # elif isinstance(module, nn.BatchNorm2d):
    #     return compute_BatchNorm2d_madd(module, inp, out)
    # elif isinstance(module, nn.MaxPool2d):
    #     return compute_MaxPool2d_madd(module, inp, out)
    # elif isinstance(module, nn.AvgPool2d):
    #     return compute_AvgPool2d_madd(module, inp, out)
    # elif isinstance(module, (nn.ReLU, nn.ReLU6)):
    #     return compute_ReLU_madd(module, inp, out)
    # elif isinstance(module, nn.Softmax):
    #     return compute_Softmax_madd(module, inp, out)
    # elif isinstance(module, nn.Linear):
    #     return compute_Linear_madd(module, inp, out)
    # elif isinstance(module, nn.Bilinear):
    #     return compute_Bilinear_madd(module, inp[0], inp[1], out)
    # else:
    #     print(f"[MAdd]: {type(module).__name__} is not supported!")
    #     return 0
    return 0

def compute_flops(module, inp, out):
    return 0

def compute_memory(module, inp, out):
    return [0,0]

class ModelHook(object):
    def __init__(self, model, input_size):
        assert isinstance(model, nn.Module)
        assert isinstance(input_size, (list, tuple))

        self._model = model
        self._input_size = input_size
        self._origin_call = dict()  # sub module call hook

        self._hook_model()
        x = torch.rand(1, *self._input_size)  # add module duration time
        self._model.eval()
        self._model(x)

    @staticmethod
    def _register_buffer(module):
        assert isinstance(module, nn.Module)

        if len(list(module.children())) > 0:
            return

        module.register_buffer('input_shape', torch.zeros(3).int())
        module.register_buffer('output_shape', torch.zeros(3).int())
        module.register_buffer('parameter_quantity', torch.zeros(1).int())
        module.register_buffer('inference_memory', torch.zeros(1).long())
        module.register_buffer('MAdd', torch.zeros(1).long())
        module.register_buffer('duration', torch.zeros(1).float())
        module.register_buffer('Flops', torch.zeros(1).long())
        module.register_buffer('Memory', torch.zeros(2).long())

    def _sub_module_call_hook(self):
        def wrap_call(module, *input, **kwargs):
            assert module.__class__ in self._origin_call

            # Itemsize for memory
            itemsize = input[0].detach().numpy().itemsize

            start = time.time()
            output = self._origin_call[module.__class__](module, *input, **kwargs)
            end = time.time()
            module.duration = torch.from_numpy(
                np.array([end - start], dtype=np.float32))

            module.input_shape = torch.from_numpy(
                np.array(input[0].size()[1:], dtype=np.int32))
            module.output_shape = torch.from_numpy(
                np.array(output.size()[1:], dtype=np.int32))

            parameter_quantity = 0
            # iterate through parameters and count num params
            for name, p in module._parameters.items():
                parameter_quantity += (0 if p is None else torch.numel(p.data))
            module.parameter_quantity = torch.from_numpy(
                np.array([parameter_quantity], dtype=np.long))

            inference_memory = 1
            for s in output.size()[1:]:
                inference_memory *= s
            # memory += parameters_number  # exclude parameter memory
            inference_memory = inference_memory * 4 / (1024 ** 2)  # shown as MB unit
            module.inference_memory = torch.from_numpy(
                np.array([inference_memory], dtype=np.float32))

            if len(input) == 1:
                madd = compute_madd(module, input[0], output)
                flops = compute_flops(module, input[0], output)
                Memory = compute_memory(module, input[0], output)
            elif len(input) > 1:
                madd = compute_madd(module, input, output)
                flops = compute_flops(module, input, output)
                Memory = compute_memory(module, input, output)
            else:  # error
                madd = 0
                flops = 0
                Memory = (0, 0)
            module.MAdd = torch.from_numpy(
                np.array([madd], dtype=np.int64))
            module.Flops = torch.from_numpy(
                np.array([flops], dtype=np.int64))
            Memory = np.array(Memory, dtype=np.int32) * itemsize
            module.Memory = torch.from_numpy(Memory)

            return output

        for module in self._model.modules():
            if len(list(module.children())) == 0 and module.__class__ not in self._origin_call:
                self._origin_call[module.__class__] = module.__class__.__call__
                module.__class__.__call__ = wrap_call

    def _hook_model(self):
        self._model.apply(self._register_buffer)
        self._sub_module_call_hook()

    @staticmethod
    def _retrieve_leaf_modules(model):
        leaf_modules = []
        for name, m in model.named_modules():
            if len(list(m.children())) == 0:
                leaf_modules.append((name, m))
        return leaf_modules

    def retrieve_leaf_modules(self):
        return OrderedDict(self._retrieve_leaf_modules(self._model))

def get_parent_node(root_node, stat_node_name):
    assert isinstance(root_node, StatNode)

    node = root_node
    names = stat_node_name.split('.')
    for i in range(len(names) - 1):
        node_name = '.'.join(names[0:i+1])
        child_index = node.find_child_index(node_name)
        assert child_index != -1
        node = node.children[child_index]
    return node


def convert_leaf_modules_to_stat_tree(leaf_modules):
    assert isinstance(leaf_modules, OrderedDict)

    create_index = 1
    root_node = StatNode(name='root', parent=None)
    for leaf_module_name, leaf_module in leaf_modules.items():
        names = leaf_module_name.split('.')
        for i in range(len(names)):
            create_index += 1
            stat_node_name = '.'.join(names[0:i+1])
            parent_node = get_parent_node(root_node, stat_node_name)
            node = StatNode(name=stat_node_name, parent=parent_node)
            parent_node.add_child(node)
            if i == len(names) - 1:  # leaf module itself
                input_shape = leaf_module.input_shape.numpy().tolist()
                output_shape = leaf_module.output_shape.numpy().tolist()
                node.input_shape = input_shape
                node.output_shape = output_shape
                node.parameter_quantity = leaf_module.parameter_quantity.numpy()[0]
                node.inference_memory = leaf_module.inference_memory.numpy()[0]
                node.MAdd = leaf_module.MAdd.numpy()[0]
                node.Flops = leaf_module.Flops.numpy()[0]
                node.duration = leaf_module.duration.numpy()[0]
                node.Memory = leaf_module.Memory.numpy().tolist()
    return StatTree(root_node)

def convert_leaf_modules_to_graph(leaf_modules):
    assert isinstance(leaf_modules, OrderedDict)

    create_index = 1
    root_node = StatNode(name='root', parent=None)
    for leaf_module_name, leaf_module in leaf_modules.items():
        names = leaf_module_name.split('.')
        for i in range(len(names)):
            create_index += 1
            stat_node_name = '.'.join(names[0:i+1])
            parent_node = get_parent_node(root_node, stat_node_name)
            label = names[i] if i == len(names) - 1 else  stat_node_name
            node = StatNode(name=label, parent=parent_node)
            # node.add_context(parent_node.name)
            parent_node.add_child(node)
            if i == len(names) - 1:  # leaf module itself
                input_shape = leaf_module.input_shape.numpy().tolist()
                output_shape = leaf_module.output_shape.numpy().tolist()
                node.input_shape = input_shape
                node.output_shape = output_shape
                node.parameter_quantity = leaf_module.parameter_quantity.numpy()[0]
                node.inference_memory = leaf_module.inference_memory.numpy()[0]
                node.MAdd = leaf_module.MAdd.numpy()[0]
                node.Flops = leaf_module.Flops.numpy()[0]
                node.duration = leaf_module.duration.numpy()[0]
                node.Memory = leaf_module.Memory.numpy().tolist()
    tree = StatTree(root_node)
    tree.plot(path="./logs/module_1.pdf")
    return tree

class ModelStat(object):
    def __init__(self, model, input_size, query_granularity=1):
        assert isinstance(model, nn.Module)
        assert isinstance(input_size, (tuple, list)) and len(input_size) == 3
        self._model = model
        self._input_size = input_size
        self._query_granularity = query_granularity

    def _analyze_model(self):
        model_hook = ModelHook(self._model, self._input_size)
        leaf_modules = model_hook.retrieve_leaf_modules()
        stat_tree = convert_leaf_modules_to_stat_tree(leaf_modules)
        collected_nodes = stat_tree.get_collected_stat_nodes(self._query_granularity)
        return collected_nodes

    def show_report(self):
        collected_nodes = self._analyze_model()
        report = report_format(collected_nodes)
        print(report)
    
    def plot_graph(self):
        model_hook = ModelHook(self._model, self._input_size)
        leaf_modules = model_hook.retrieve_leaf_modules()
        stat_tree = convert_leaf_modules_to_graph(leaf_modules)
        # collected_nodes = stat_tree.get_collected_stat_nodes(self._query_granularity)

        return 


def module_stat(model, input_size, query_granularity=1):
    ms = ModelStat(model, input_size, query_granularity)
    # ms.show_report()
    ms.plot_graph()
