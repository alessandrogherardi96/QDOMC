import random
import networkx as nx
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy
import matplotlib.pyplot as plt
import pickle
from .basic import func_max

def mc_upper_bound(G):
	"""
	INPUT:
	 - "G" Networkx Undirected Graph
	OUTPUT:
	 - "chromatic_number" integer upper bound on the maximum clique number
	"""
	answ = nx.algorithms.coloring.greedy_color(G)
	chromatic_number = list(set(list(answ.values())))
	return len(chromatic_number)

def random_node(nodes, v, no_node_list):
    """
    INPUT:
     - "nodes" list: list of nodes in the graph
     - "v" integer: node that should be avoided
     - "no_node_list" list: nodes that should not be chosen
	OUTPUT:
	 - A random node that is neither 'v' nor in the 'no_node_list'. Returns None if all nodes are excluded.

    This recursive function selects a random node from the list of available nodes. If all nodes 
    are excluded by the given conditions, it returns None to stop the process.
    """
    node = random.choice(list(nodes))
    if len(no_node_list) >= len(nodes)-1:
        return None
    elif node != v and node not in no_node_list:
        return node
    else:
        return random_node(nodes, v, no_node_list)
    
def create_graph_with_rand_clique(num_nodes, dim_cli, num_cli, random_clique_dimension):
    """
    INPUT:
     - "num_nodes" integer: total number of nodes in the graph
     - "dim_cli" integer: number of nodes in each clique
     - "num_cli" integer: number of cliques to generate
     - "random_clique_dimension" boolean: whether to randomize the clique size
    OUTPUT:
     - A graph with specified cliques, the list of clique nodes, and the remaining nodes.

    This function generates a graph with 'num_nodes' nodes and creates 'num_cli' cliques. 
    Each clique contains 'dim_cli' nodes, and edges are added to form complete cliques. 
    The size of the cliques can be fixed or randomized.
    """
    g = nx.Graph()
    cliques_nodes = []
    for i in range(num_nodes):
        g.add_node(i)
    nodes = list(g.nodes())
    for k in range(num_cli):
        if random_clique_dimension:
            clique = random.sample(nodes, random.randint(2,dim_cli))
        else:
            clique = random.sample(nodes, dim_cli)
        cliques_nodes.append(sorted(clique))
        nodes = [a for a in nodes if a not in clique]
        combinations = itertools.combinations(clique,2)
        g.add_edges_from(combinations)
    return g, cliques_nodes, nodes

def graph_with_clique_density(num_node, ex_node, num_clique, density, add_edges_between_cliques, random_clique_dimension):
    """
    INPUT:
     - "num_node" integer: number of nodes per clique
     - "ex_node" integer: number of extra nodes outside the cliques
     - "num_clique" integer: number of cliques in the graph
     - "density" integer: density of the graph, used to define edge connections
     - "add_edges_between_cliques" boolean: if True, adds edges between cliques
     - "random_clique_dimension" boolean: whether to randomize the size of cliques
    OUTPUT:
     - A graph with cliques and a list of nodes in clique, with edges based on the provided density.

    This function generates a graph with 'num_clique' cliques, each containing 'num_node' nodes. 
    It connects the extra nodes (specified by 'ex_node') to the cliques while maintaining 
    an upper bound on the clique number (using mc_upper_bound), based on density value. Optionally, it adds edges between 
    the cliques.
    """
    g, clique_nodes, list_ex_nodes = create_graph_with_rand_clique(num_node*num_clique+ex_node, num_node, num_clique, random_clique_dimension)
    total_nodes = len(g.nodes())
    ub = mc_upper_bound(g)
    ld = (density-5)/100 #lower density
    ud = (density+5)/100 #upper density
    for i in tqdm(list_ex_nodes, desc="Connecting extra nodes", unit="node"):
        links = random.randint(int(total_nodes*ld), int(total_nodes*ud)) #number of iterations
        no_nodes_1 = []
        for _ in range(links):
            node = random_node(g.nodes, i, no_nodes_1) #Chose node
            if node is None: #If all node are used, break
                break
            g.add_edge(i, node) #Add edge
            new_ub = mc_upper_bound(g) #Caculate new upper bound
            if new_ub > ub: #Remove edge if necessary
                g.remove_edge(i, node)
            no_nodes_1.append(node) #Add node to no node list
    if add_edges_between_cliques:
        counter = 1
        for cli in clique_nodes:
            print(f"Connecting clique {counter} of {len(clique_nodes)}")
            counter = counter + 1
            ok_nodes = [a for a in g.nodes() if a not in cli]
            for n in tqdm(cli, desc="Connecting clique node", unit="node"):
                links = random.randint(int(total_nodes*ld), int(total_nodes*ud)) #Number of iterations
                no_nodes_2 = []
                for _ in range(links):
                    node = random_node(ok_nodes, n, no_nodes_2) #Chose node
                    if node is None: #If all node are used, break
                        break
                    g.add_edge(n, node) #Add edge
                    new_ub = mc_upper_bound(g) #Caculate new upper bound
                    if new_ub > ub: #Remove edge if necessary
                        g.remove_edge(n, node)
                    no_nodes_2.append(node) #Add node to no node list
    return g, clique_nodes

def graph_with_clique_degree(num_node, ex_node, num_clique, degree_min, degree_max, add_edges_between_cliques, random_clique_dimension):
    """
    INPUT:
     - "num_node" integer: number of nodes per clique
     - "ex_node" integer: number of extra nodes outside the cliques
     - "num_clique" integer: number of cliques in the graph
     - "degree_min" float: minimum value of the ratio node_degree/total_nodes
     - "degree_max" float: maximum value of the ratio node_degree/total_nodes
     - "add_edges_between_cliques" boolean: if True, adds edges between cliques
     - "random_clique_dimension" boolean: whether to randomize the size of cliques
    OUTPUT:
     - A graph with cliques and a list of nodes in clique, with edges added based on the provided values degree_min and degree_max.

    This function generates a graph with 'num_clique' cliques, each containing 'num_node' nodes. 
    It connects the extra nodes (specified by 'ex_node') to the cliques while maintaining 
    an upper bound on the clique number (using mc_upper_bound). Connections are made so that the ratio node_degree/total_nodes
    remains between degree_min and degree_max value. Optionally, it adds edges between the cliques.
    """
    g, clique_nodes, list_ex_nodes = create_graph_with_rand_clique(num_node*num_clique+ex_node, num_node, num_clique, random_clique_dimension)
    total_nodes = len(g.nodes())
    ub = mc_upper_bound(g)
    for i in tqdm(list_ex_nodes, desc="Connecting extra nodes", unit="node"):
        node_degree = random.uniform(degree_min, degree_max) #Chose random degree
        no_nodes_1 = []
        while (g.degree()[i]/total_nodes) < node_degree: #While ratio below node_degree, the function continue add adges
            node = random_node(g.nodes, i, no_nodes_1) #Chose node
            if node is None: #If all node are used, break
                break
            g.add_edge(i, node) #Add edge
            new_ub = mc_upper_bound(g) #Caculate new upper bound
            if new_ub > ub: #Remove edge if necessary
                g.remove_edge(i, node)
            no_nodes_1.append(node) #Add node to no node list
    if add_edges_between_cliques:
        counter = 1
        for cli in clique_nodes:
            print(f"Connecting clique {counter} of {len(clique_nodes)}")
            counter = counter + 1
            ok_nodes = [a for a in g.nodes() if a not in cli]
            for n in tqdm(cli, desc="Connecting nodes in cliques", unit="node"):
                node_degree = random.uniform(degree_min, degree_max) #Chose random degree
                no_nodes_2 = []
                while (g.degree()[n]/total_nodes)< node_degree: #While ratio below node_degree, the function continue add adges
                    node = random_node(ok_nodes, n, no_nodes_2) #Chose node
                    if node is None: #If all node are used, break
                        break
                    g.add_edge(n, node) #Add edge
                    new_ub = mc_upper_bound(g) #Caculate new upper bound
                    if new_ub > ub: #Remove edge if necessary
                        g.remove_edge(n, node)
                    no_nodes_2.append(node) #Add node to no node list
    return g, clique_nodes

def create_range(a, b):
    """
    Generate a list of values starting from `a` and incrementing randomly 
    until reaching `a + b`. The function ensures at least one large increment 
    is used during the generation process.

    Parameters:
    - a (int): The starting value for the range.
    - b (int): The length to add to `a` to determine the stopping point.

    Returns:
    - list: A list of integers starting from `a`, with random increments 
      until reaching `a + b`.
    """
    # Initialize variables
    values = []
    current_sum = a
    used_large_increment = False
    # Generate values until the total reaches `a + b`
    while current_sum < a + b:
        if not used_large_increment and random.random() < 1:
            # Generate a large increment (> a)
            increment = random.randint(a + 2, a + int(2 * a / 3))
            used_large_increment = True
        else:
            # Generate a smaller increment
            increment = random.randint(3, a // 2)
        current_sum += increment 
        # Ensure the sum doesn't exceed `a + b`
        if current_sum >= a + b:
            current_sum = a + b
        # Add the current sum to the list
        values.append(current_sum)
    return values

def set_weights(g, clique, val):
    """
    This function adjusts the weights of nodes in a graph `g` by assigning random values to attributes `w1`, `w2`, `w3`, `w4` 
    and ensures that the maximum value of a specific function `func_max` is achieved for a given clique (clique 0).

    Parameters:
    - g: The input graph (assumed to be a NetworkX graph).
    - clique: The size of the clique for which weights should be maximum. Must be the first clique with nodes 0 to K.
    - val: A list of integers that indicate the ranges of cliques (es: [10,20,30,35]).

    Returns:
    - g: The graph with updated node attributes.
    - func_values: A list of function values corresponding to the calculated subsets.

    The function uses random values within specific ranges to set weights for the nodes.
    If the maximum value of `func_max` is not associated with the clique, the weights are recalculated.
    The process continues iteratively until the condition is satisfied.
    """
    count = 0
    while True:
        g_copy = g.copy()
        attributes = {}
        for i in range(len(g.nodes())):    
            w1 = random.uniform(0,0.027290137282490516)
            w2 = random.uniform(0,0.03640364796375065)
            w3 = random.uniform(0,0.04098560068502567)
            w4 = random.uniform(0,0.04098560068502567)
            if i in range(clique) and count > 500:
                w1 = random.uniform(0.027290137282490516/2,0.027290137282490516)
                w2 = random.uniform(0.03640364796375065/2,0.03640364796375065)
                w3 = random.uniform(0.04098560068502567/2,0.04098560068502567)
                w4 = random.uniform(0.04098560068502567/2,0.04098560068502567)
            attr_dict = {'w1':w1, 'w2':w2, 'w3':w3, 'w4':w4}
            attributes[i]=attr_dict
        nx.set_node_attributes(g_copy, attributes)
        func_values = [func_max(g_copy, [i for i in range(clique)])]
        for nod in range(len(val)):
            if nod == 0:
                func_values.extend([func_max(g_copy, [i for i in range(clique, val[nod])])])
            else:
                func_values.extend([func_max(g_copy, [i for i in range(val[nod-1], val[nod])])])
        if max(func_values) == func_values[0]:
            break
        count = count+1
    nx.set_node_attributes(g, attributes)
    return g, func_values

def create_benchmark(clique, other):
    """
    This function creates a benchmark graph composed of a main clique connected to additional cliques via bridges. 
    Each clique is represented as a subset of nodes, and weights are assigned to optimize a specific function for the cliques.

    Parameters:
    - clique: The size of the main clique.
    - other: The total number of additional nodes outside the main clique.

    Returns:
    - g: The generated graph (NetworkX graph) with nodes and edges.
    - dict: A dictionary where:
        - Keys are indices of cliques.
        - Values are tuples containing the clique's nodes and the corresponding function values.

    The function builds the graph by:
    1. Generating a main clique.
    2. Creating additional cliques connected to the main clique by bridge edges.
    3. Assigning weights to nodes using the `set_weights` function.
    4. Calculating function values for each clique and storing them in the output dictionary.
    """
    g = nx.Graph()
    nodes = [i for i in range(clique+other)]
    edges = [(i,j) for i in range(clique) for j in range(clique) if i<j]
    val = create_range(clique, other)
    cliques = [[i for i in range(clique)]]
    for nod in range(len(val)):
        if nod == 0:
            e = [(i,j) for i in range(clique, val[nod]) for j in range(clique, val[nod]) if i<j]
            bridge = [(clique-1, clique)]
            cliques.append([i for i in range(clique, val[nod])])
            edges.extend(e)
            edges.extend(bridge)
        else:
            e = [(i,j) for i in range(val[nod-1], val[nod]) for j in range(val[nod-1], val[nod]) if i<j]
            bridge = [(val[nod-1]-1, val[nod-1])]
            cliques.append([i for i in range(val[nod-1], val[nod])])
            edges.extend(e)
            if bridge[0] not in edges:
                edges.extend(bridge)
    for i in nodes:
        g.add_node(i)
    g.add_edges_from(edges)
    g, function_values = set_weights(g, clique, val)
    dict = {}
    for i in range(len(cliques)):
        dict[i] = [cliques[i], function_values[i]]
    return g, dict