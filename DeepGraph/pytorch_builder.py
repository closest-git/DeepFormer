"""
HiddenLayer

PyTorch graph importer.
 
Written by Waleed Abdulla
Licensed under the MIT License
"""

from __future__ import absolute_import, division, print_function
import re
from .task_graph import *
from graphviz import Digraph
# from . import transforms as ht
import torch
import torchvision
from torch.autograd import Variable
import warnings
from distutils.version import LooseVersion
# PyTorch Graph Transforms
# FRAMEWORK_TRANSFORMS = [
#     # Hide onnx: prefix
#     ht.Rename(op=r"onnx::(.*)", to=r"\1"),
#     # ONNX uses Gemm for linear layers (stands for General Matrix Multiplication).
#     # It's an odd name that noone recognizes. Rename it. 
#     ht.Rename(op=r"Gemm", to=r"Linear"),
#     # PyTorch layers that don't have an ONNX counterpart
#     ht.Rename(op=r"aten::max\_pool2d\_with\_indices", to="MaxPool"),
#     # Shorten op name
#     ht.Rename(op=r"BatchNormalization", to="BatchNorm"),
# ]


def dump_pytorch_graph(graph):
    """List all the nodes in a PyTorch graph."""
    f = "{:25} {:40}   {} -> {}"
    print(f.format("kind", "scopeName", "inputs", "outputs"))
    for node in graph.nodes():
        print(f.format(node.kind(), node.scopeName(),
                       [i.unique() for i in node.inputs()],
                       [i.unique() for i in node.outputs()]
                       ))


def pytorch_id(node):
    """Returns a unique ID for a node."""
    # After ONNX simplification, the scopeName is not unique anymore
    # so append node outputs to guarantee uniqueness
    return node.scopeName() + "/outputs/" + "/".join(["{}".format(o.unique()) for o in node.outputs()])


def get_shape(torch_node):
    """Return the output shape of the given Pytorch node."""
    # Extract node output shape from the node string representation
    # This is a hack because there doesn't seem to be an official way to do it.
    # See my quesiton in the PyTorch forum:
    # https://discuss.pytorch.org/t/node-output-shape-from-trace-graph/24351/2
    # TODO: find a better way to extract output shape
    # TODO: Assuming the node has one output. Update if we encounter a multi-output node.
    m = re.match(r".*Float\(([\d\s\,]+)\).*", str(next(torch_node.outputs())))
    if m:
        shape = m.group(1)
        shape = shape.split(",")
        shape = tuple(map(int, shape))
    else:
        shape = None
    return shape


def get_fn_name(fn, show_attrs, max_attr_chars):
    name = str(type(fn).__name__)
    if not show_attrs:
        return name
    attrs = dict()
    for attr in dir(fn):
        if not attr.startswith(SAVED_PREFIX):
            continue
        val = getattr(fn, attr)
        attr = attr[len(SAVED_PREFIX):]
        if torch.is_tensor(val):
            attrs[attr] = "[saved tensor]"
        elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
            attrs[attr] = "[saved tensors]"
        else:
            attrs[attr] = str(val)
    if not attrs:
        return name
    max_attr_chars = max(max_attr_chars, 3)
    col1width = max(len(k) for k in attrs.keys())
    col2width = min(max(len(str(v)) for v in attrs.values()), max_attr_chars)
    sep = "-" * max(col1width + col2width + 2, len(name))
    attrstr = '%-' + str(col1width) + 's: %' + str(col2width)+ 's'
    truncate = lambda s: s[:col2width - 3] + "..." if len(s) > col2width else s
    params = '\n'.join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
    return name + '\n' + sep + '\n' + params

def resize_graph(dot, size_per_element=0.15, min_size=12):
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)

#  Generates graph using the backwards pass(The operator names are taken from the backward pass, so some of them are difficult to understand)
def grad2Graph(graph,model,input,show_attrs=False, show_saved=False, max_attr_chars=50,transforms=None):    
    if LooseVersion(torch.__version__) < LooseVersion("1.9") and \
        (show_attrs or show_saved):
        warnings.warn(
            "make_dot: showing grad_fn attributes and saved variables"
            " requires PyTorch version >= 1.9. (This does NOT apply to"
            " saved tensors saved by custom autograd functions.)")
    
    y = model(input)
    var = y
    params=dict(model.named_parameters())
    # dot = make_dot(y.mean(), params=dict(model.named_parameters()))

    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}
    else:
        param_map = {}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='10',
                     ranksep='0.1',
                     height='0.2',
                     fontname='monospace')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def get_var_name(var, name=None):
        if not name:
            name = param_map[id(var)] if id(var) in param_map else ''
        return '%s\n %s' % (name, size_to_str(var.size()))

    def add_nodes(fn):
        assert not torch.is_tensor(fn)
        if fn in seen:
            return
        seen.add(fn)

        if show_saved:
            for attr in dir(fn):
                if not attr.startswith(SAVED_PREFIX):
                    continue
                val = getattr(fn, attr)
                seen.add(val)
                attr = attr[len(SAVED_PREFIX):]
                if torch.is_tensor(val):
                    dot.edge(str(id(fn)), str(id(val)), dir="none")
                    dot.node(str(id(val)), get_var_name(val, attr), fillcolor='orange')
                if isinstance(val, tuple):
                    for i, t in enumerate(val):
                        if torch.is_tensor(t):
                            name = attr + '[%s]' % str(i)
                            dot.edge(str(id(fn)), str(id(t)), dir="none")
                            dot.node(str(id(t)), get_var_name(t, name), fillcolor='orange')

        if hasattr(fn, 'variable'):
            # if grad_accumulator, add the node for `.variable`
            var = fn.variable
            seen.add(var)
            dot.node(str(id(var)), get_var_name(var), fillcolor='lightblue')
            dot.edge(str(id(var)), str(id(fn)))

        # add the node for this grad_fn
        dot.node(str(id(fn)), get_fn_name(fn, show_attrs, max_attr_chars))

        # recurse
        if hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    dot.edge(str(id(u[0])), str(id(fn)))
                    add_nodes(u[0])

        # note: this used to show .saved_tensors in pytorch0.2, but stopped
        # working* as it was moved to ATen and Variable-Tensor merged
        # also note that this still works for custom autograd functions
        if hasattr(fn, 'saved_tensors'):
            for t in fn.saved_tensors:
                dot.edge(str(id(t)), str(id(fn)))
                dot.node(str(id(t)), get_var_name(t), fillcolor='orange')


    def add_base_tensor(var, color='darkolivegreen1'):
        if var in seen:
            return
        seen.add(var)
        dot.node(str(id(var)), get_var_name(var), fillcolor=color)
        if (var.grad_fn):
            add_nodes(var.grad_fn)
            dot.edge(str(id(var.grad_fn)), str(id(var)))
        if var._is_view():
            add_base_tensor(var._base, color='darkolivegreen3')
            dot.edge(str(id(var._base)), str(id(var)), style="dotted")


    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_base_tensor(v)
    else:
        add_base_tensor(var)

    resize_graph(dot)
    dot.format = "pdf"
    dot.render("./grad2graph.pdf",cleanup=True)
    return dot

def trace2graph(graph, model, args, input_names=None, verbose=False):
    ver = LooseVersion(torch.__version__)

    # Run the Pytorch graph to get a trace and generate a graph from it
    # torch.jit.trace (does not always work)!!!     https://discuss.pytorch.org/t/difference-in-get-trace-graph-in-version-1-7-and-get-trace-graph-in-version-1-1/105914            
    try:
        from torch.jit import _get_trace_graph as trace_graph
        trace, out = trace_graph(model, args) 
        torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    except ImportError:
        try:
            from torch.jit import get_trace_graph as trace_graph
            trace, out = trace_graph(model, args) 
            exType = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK      #torch.onnx.OperatorExportTypes.ONNX
            torch.onnx._optimize_trace(trace,exType )       #   !!!    Unsupported prim::Constant kind: `s`. Send a bug report.
            torch_graph = trace.graph()
        except ImportError:
            raise ValueError("module 'torch.jit' has no attribute 'get_trace_graph'.")

    

    # Dump list of nodes (DEBUG only)
    if True:
        dump_pytorch_graph(torch_graph)

    # Loop through nodes and build HL graph
    for torch_node in torch_graph.nodes():
        # Op
        op = torch_node.kind()
        # Parameters
        params = {k: torch_node[k] for k in torch_node.attributeNames()} 
        # Inputs/outputs
        # TODO: inputs = [i.unique() for i in node.inputs()]
        outputs = [o.unique() for o in torch_node.outputs()]
        # Get output shape
        shape = get_shape(torch_node)
        # Add HL node
        hl_node = Node(uid=pytorch_id(torch_node), name=None, op=op,output_shape=shape, params=params)
        graph.add_node(hl_node)
        # Add edges
        for target_torch_node in torch_graph.nodes():
            target_inputs = [i.unique() for i in target_torch_node.inputs()]
            if set(outputs) & set(target_inputs):
                graph.add_edge_by_id(pytorch_id(torch_node), pytorch_id(target_torch_node), shape)
    return graph

def build_graph(model=None, args=None, input_names=None,transforms="default", framework_transforms="default",path=""):
    # Initialize an empty graph
    g = TaskGraph()

    # Detect framwork
    framework = detect_framework(model)
    assert framework == "torch"
    assert args is not None, "Argument args must be provided for Pytorch models."
    if False:
        trace2graph(g, model, args)      # many bugs!
    else:
        grad2Graph(g, model, args) 
    
    # Apply Transforms
    # if framework_transforms:
    #     if framework_transforms == "default":
    #         framework_transforms = FRAMEWORK_TRANSFORMS
    #     for t in framework_transforms:
    #         g = t.apply(g)
    # if transforms:
    #     if transforms == "default":
    #         from .transforms import SIMPLICITY_TRANSFORMS
    #         transforms = SIMPLICITY_TRANSFORMS
    #     for t in transforms:
    #         g = t.apply(g)
    return g

if __name__ == "__main__":
    # model = torchvision.models.resnet101()
    model = torchvision.models.resnet18()

    # Rather than using the default transforms, build custom ones to group
    # nodes of residual and bottleneck blocks.
    # transforms = [
    #     # Fold Conv, BN, RELU layers into one
    #     hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
    #     # Fold Conv, BN layers together
    #     hl.transforms.Fold("Conv > BatchNorm", "ConvBn"),
    #     # Fold bottleneck blocks
    #     hl.transforms.Fold("""
    #         ((ConvBnRelu > ConvBnRelu > ConvBn) | ConvBn) > Add > Relu
    #         """, "BottleneckBlock", "Bottleneck Block"),
    #     # Fold residual blocks
    #     hl.transforms.Fold("""ConvBnRelu > ConvBnRelu > ConvBn > Add > Relu""",
    #                     "ResBlock", "Residual Block"),
    #     # Fold repeated blocks
    #     hl.transforms.FoldDuplicates(),
    # ]

    # Display graph using the transforms above
    g = build_graph(model, torch.zeros([1, 3, 224, 224]),transforms=None)
    # g.theme = g.graph.THEMES["blue"].copy()
    g.save("./1.pdf")