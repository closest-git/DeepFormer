import numbers
from graphviz import Digraph
from torch.autograd import Variable
# from ..core.node import Variable


def plot_comp_graph(top_node, view=False, name="comp_graph"):
    print("\nPlotting...")
    graph = XDigraph("Computational graph", filename=name, engine="dot")
    graph.attr(size="6,6")
    graph.node_attr.update(color='lightblue2', style="filled")
    graph.graph_attr.update(rankdir="BT")

    graph.add_node_subgraph_to_plot_graph(top_node)

    graph.render(view=view, cleanup=True)


class XDigraph(Digraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # print(self.node_attr)
        self.node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='10',
                        ranksep='0.1',
                        height='0.2',
                        fontname='monospace')
        # self.attr(size="12,12")
        # print(self.node_attr)
        self.added_nodes = set()
    
    def resize_graph(self, size_per_element=0.15, min_size=12):
        num_rows = len(self.body)
        content_size = num_rows * size_per_element
        size = max(min_size, content_size)
        size_str = str(size) + "," + str(size)
        self.graph_attr.update(size=size_str)
        print(self.graph_attr)

    @staticmethod
    def id_str(node):
        return node.name
        # return str(node.id)

    def add_node(self, node, root_graph=None):
        if root_graph is None:
            root_graph = self
        label = node.name           #str(node)
        super().node(XDigraph.id_str(node),
                     label=label,
                     color=XDigraph.get_color(node),
                     shape=XDigraph.get_shape(node))
        root_graph.added_nodes.add(XDigraph.id_str(node))

    def add_edge(self, child, parent,style={"style": "filled"}):       
        self.edge(XDigraph.id_str(child),
                  XDigraph.id_str(parent),
                  **style)

    @staticmethod
    def get_color(node):
        if isinstance(node, Variable):
            # better way to figure out the coloring?
            if isinstance(node.value, numbers.Number) and node.value == 1 and node.name[-4:] == "grad":
                return "gray"
            return "indianred1"
        else:
            return "lightblue"

    @staticmethod
    def get_shape(node):
        if isinstance(node, Variable):
            return "box"
        else:
            return "oval"

    def add_node_with_context(self, node, ctx, root_graph=None):
        """
        Add just the node (not the connections, not the children) to the respective subgraph
        """
        if root_graph is None:
            root_graph = self
        if len(ctx):
            with self.subgraph(name="cluster" + ctx[0]) as subgraph:
                subgraph.attr(color="blue")
                subgraph.attr(label=ctx[0].split("_")[0])

                subgraph.add_node_with_context(node, ctx[1:], root_graph=self)
        else:
            self.add_node(node, root_graph)

    def add_node_subgraph_to_plot_graph(self, top_node):
        if XDigraph.id_str(top_node) not in self.added_nodes:
            self.add_node_with_context(top_node, top_node.context_list)

            for i,child in enumerate(top_node.children):
                child.add_context(top_node.name,top_node.context_list)
                if i > 0:            self.add_edge(top_node.children[i-1],child)
                # self.add_edge(child, top_node)
            if len(top_node.children)>0:
                self.add_edge(child, top_node,{"style":"filled","dir":"both", "arrowhead":"diamond",  "arrowtail":"diamond"})

            # Make each of the children do the same, but skip duplicates
            for child in set(top_node.children):
                self.add_node_subgraph_to_plot_graph(child)
        
