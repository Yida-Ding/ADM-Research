import networkx as nx



G = nx.Graph()
nx.add_path(G, [0, 1, 2, 3])
nx.add_path(G, [10, 11, 12])
print(G.nodes)
G.add_edge()
