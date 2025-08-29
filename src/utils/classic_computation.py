from .basic import mc_upper_bound
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

def find_bigger_graph(g_or, start, dicts, values):
    """
    Input:
        g_or (networkx.Graph): The original NetworkX graph.
        start (list): A list of starting nodes that must remain in the graph.
        dicts (list of dict): The list of weight dictionaries to be used as a filter
        values (list): Minimum values below which the weight W_i cannot fall, coupled with dicts
        Example: find_bigger_graph(G, start_point, [w1,w2,w3], [w1_max, w2_max, 100]) --> keeps nodes with maximum w1 and w2 and w3>w3_max/100
    
    Output:
        networkx.Graph: A reduced subgraph of the original graph.
    
    Explanation:
        This function creates a subgraph of the original graph by:
        1. Removing nodes that do not meet the relative threshold condition across the given `dicts` and `values`.
        2. Computing the set of nodes that are common neighbors of all nodes in `start`.
        3. Keeping only these common neighbors (plus the `start` nodes themselves) in the graph.
        The result is a filtered subgraph containing only the relevant nodes.
    """
    g = g_or.copy()
    max_values = [max(d.values()) for d in dicts]
    if dicts and values:
        for k in dicts[0]:
            if any(dicts[i][k] < max_values[i] / values[i] for i in range(len(dicts))):
                g.remove_node(k)
    all_neigh = []
    first = True
    for n in start:
        if first:
            all_neigh.extend(list(nx.neighbors(g, n)))
            first = False
        else:
            all_neigh = list(set(all_neigh).intersection(list(nx.neighbors(g,n))))
    all_neigh.extend(start)
    for node in list(g.nodes()):
        if node not in all_neigh:
            g.remove_node(node)
    return g

def remove_nodes_ub(g_or, start, dict):
    """
    Input:
        g_or (networkx.Graph): The original NetworkX graph.
        start (list): A list of nodes that must not be removed.
        dict (dict): The dictionary of the weight of nodes to bne used for removing nodes (example: w1 or w2)
    
    Output:
        networkx.Graph: A reduced graph after removing certain nodes.
    
    Explanation:
        This function iteratively attempts to remove nodes from the graph that are not in `start`.
        Nodes are considered in ascending order of their values in `dict`. 
        A node is removed only if the upper bound of the graph (computed by `mc_upper_bound`) does not decrease after its removal. 
        The resulting graph contains only nodes that are necessary to maintain the same upper bound.
    """
    g = g_or.copy()
    ub = mc_upper_bound(g)
    nodes = [n for n, _ in sorted(dict.items(), key=lambda x: x[1]) if n in g]
    for i in tqdm(nodes):
        if i not in start:
            g_rem = g.copy()
            g_rem.remove_node(i)
            ub2 = mc_upper_bound(g_rem)
            if ub2 == ub:
                g.remove_node(i)
    return g

def partial_func(w1, w2, w3, w4, cli):
    """
    Input:
        w1, w2, w3, w4 (list of float): Dicts of weights.
        cli (list of int): The clique find
    
    Output:
        tuple: A tuple (t1, t2, sum4) of computed partial values.
    
    Explanation:
        This function computes partial sums of weights for the indices in `cli`.
        - sum1 to sum4 are the sums of w1, w2, w3, and w4 over `cli`.
        - t1 is a capped and shifted version of sum1 based on constants z1 and z2.
        - t2 is a capped combination of sum2 and sum3, adjusted by constants z3, z4, and z5.
        - sum4 is returned as-is.
        The function essentially transforms the weighted sums into bounded partial metrics.
    """
    sum1 = sum(w1[i] for i in cli)
    sum2 = sum(w2[i] for i in cli)
    sum3 = sum(w3[i] for i in cli)
    sum4 = sum(w4[i] for i in cli)

    z1 = 0.17310630818265274
    z2 = 0.15487928682013247
    z3 = 0.03640364796375065
    z4 = 0.027290137282490516
    z5 = 0.14576577613887232

    t1 = min(max(sum1 - z1, 0), z2)
    t2_inner = max(sum2 - z3, 0) + sum3 - z4
    t2 = min(max(t2_inner, 0), z5)

    return t1, t2, sum4

def draw_max(p_1,p_2,p_3):
    """
    Input:
        p_1, p_2, p_3 (float): Current values of the three metrics to be visualized, extracted from partial_func.
    
    Output:
        None
    
    Explanation:
        This function creates a bar chart comparing the current values (p_1, p_2, p_3) 
        against their respective maximum values. Each metric is labeled according to 
        its computation formula. The chart displays both current and maximum values 
        side by side, with numeric labels on top of each bar for clarity. 
        A grid, legend, and proper axis labeling are added for readability.
    """

    labels = ['min(max(sum_w1-a1,0),a2)', ' min(max(sum_w2-a3,0)+sum_w3-a4,0),a5)', 'sum_w4']
    values = [p_1, p_2, p_3]
    max_values = [0.15487928682013247, 0.14576577613887232, 0.24831508986982237]

    x = range(len(labels))

    plt.figure(figsize=(9,5))
    bar_width = 0.35

    bars_current = plt.bar(x, values, width=bar_width, label='Valore attuale', color='skyblue')
    bars_max = plt.bar([i + bar_width for i in x], max_values, width=bar_width, label='Valore massimo', color='orange')

    for bar in bars_current:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.001, f'{height:.4f}', ha='center', va='bottom')

    for bar in bars_max:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.001, f'{height:.4f}', ha='center', va='bottom')

    plt.xticks([i + bar_width / 2 for i in x], labels)
    plt.ylabel('Value')
    plt.title('Actual values vs maximum')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()