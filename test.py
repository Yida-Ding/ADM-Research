import networkx as nx
import numpy as np

g=nx.DiGraph()
g.add_edges_from([(1,2),(3,2)])
if 1 in g.nodes:
    g.remove_node(1)
print(g.edges)




