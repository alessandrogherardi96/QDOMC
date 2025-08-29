import networkx as nx
import numpy as np

def mc_upper_bound(g):
	"""
	INPUT:
	 - "G" Networkx Undirected Graph
	OUTPUT:
	 - "chromatic_number" integer upper bound on the maximum clique number
	"""
	answ = nx.algorithms.coloring.greedy_color(g)
	chromatic_number = list(set(list(answ.values())))
	return len(chromatic_number)

def func_max(g, cli):
    """
    This function calculates the maximum objective value for a given subset of nodes (clique) in a graph `g`, 
    based on specific node attributes and predefined thresholds.

    Parameters:
    - g: A NetworkX graph with node attributes ("w1", "w2", "w3", "w4").
    - cli: A list of nodes representing the clique for which the function value is computed.

    Returns:
    - f: A float representing the computed objective value for the given clique.

    Key steps:
    1. Retrieves the attributes ("w1", "w2", "w3", "w4") for all nodes in the graph.
    2. Defines constant thresholds (`z1`, `z2`, `z3`, `z4`, `z5`) used for the calculation.
    3. Calculates the sum of each attribute (`sum1`, `sum2`, `sum3`, `sum4`) for the nodes in the clique.
    4. Applies a mathematical formula to compute the objective value `f`:
    - Evaluates differences between attribute sums and thresholds.
    - Uses nested `min` and `max` functions to constrain and combine these differences with the attribute sums.
    5. Returns the computed value `f`.
    """
    w1 = nx.get_node_attributes(g, "w1")
    w2 = nx.get_node_attributes(g, "w2")
    w3 = nx.get_node_attributes(g, "w3")
    w4 = nx.get_node_attributes(g, "w4")
    z1 = 0.17310630818265274
    z2 = 0.15487928682013247
    z3 = 0.03640364796375065
    z4 = 0.027290137282490516
    z5 = 0.14576577613887232
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for i in cli:
        sum1+=w1[i]
        sum2+=w2[i]
        sum3+=w3[i]
        sum4+=w4[i]
    f = min(max(sum1-z1,0),z2) + min(max(max(sum2-z3,0)+sum3-z4,0),z5)+sum4
    return f

def QUBO_bias(g, b, weighted, arr):
    """
    This function calculates the QUBO (Quadratic Unconstrained Binary Optimization) bias values for nodes in a graph `g` 
    based on their attributes and certain weights. The function optionally applies bias and weighting schemes 
    to influence the optimization process.

    Parameters:
    - g: A NetworkX graph with node attributes ("w1", "w2", "w3", "w4").
    - b (bool): If True, biases are applied based on the "w4" attribute.
    - weighted (bool): If True, node costs are calculated using weighted attributes.
    - arr: A list of four values that are used as the weighting factors for the attributes "w1", "w2", "w3", and "w4" (in that order). 
        Each value in `arr` is multiplied with the corresponding attribute to scale the node costs. 

    Returns:
    - q: A dictionary representing the QUBO matrix. 
    Each key is a tuple of nodes (i, j), and the value is the bias/cost or penalty.

    Key steps:
    1. Extracts node attributes ("w1", "w2", "w3", "w4") and calculates their maximum, minimum, and median values.
    2. Defines bias values for nodes based on their "w4" attribute and predefined thresholds if `b` is True.
    3. Calculates node costs using a weighted sum of the attributes if `weighted` is True.
    4. Scales the costs and assigns diagonal QUBO values (penalizing or favoring nodes based on costs).
    5. Adds penalties for edges in the complement graph to ensure they are discouraged in the solution.
    """
    has_weights = all(all(attr in dati for attr in ["w1","w2","w3","w4"]) for _, dati in g.nodes(data=True))
    if has_weights:
        w1 = nx.get_node_attributes(g, "w1")
        w2 = nx.get_node_attributes(g, "w2")
        w3 = nx.get_node_attributes(g, "w3")
        w4 = nx.get_node_attributes(g, "w4")
        max_w1 = max(w1.values())
        max_w2 = max(w2.values())
        max_w3 = max(w3.values())
        max_w4 = max(w4.values())
        min_w4 = min(w4.values())
        med_w4 = (min_w4+max_w4)/2
        min_bias = min([i for i in w4.values() if i != 0], default=None)/2
    cost = dict()
    q = dict()
    for i in g.nodes():
        if b:
            max_bias = max_w4+ 80*max_w4/100
            if w4[i]==0:
                bias = min_bias
            elif w4[i]>0 and w4[i]<=med_w4/2:
                bias = w4[i]+w4[i]*20/100
            elif w4[i]>med_w4/2 and w4[i]<=med_w4:
                bias = w4[i]+w4[i]*40/100
            elif w4[i]>med_w4 and w4[i]>3*med_w4/2:
                bias = w4[i]+w4[i]*60/100
            else:
                bias = w4[i]+w4[i]*80/100
        else:
            bias = 0
            max_bias = 0
        if weighted:
            cost[i] = arr[0]*w1[i]/max_w1+arr[1]*w2[i]/max_w2+arr[2]*w3[i]/max_w3+arr[3]*(w4[i]+bias)/(max_w4+max_bias)
        else:
            cost[i] = 1
    scale = max(cost.values())
    q = {(node, node): min(-cost[node] / scale, 0.0) for node in g}
    comp_G = nx.complement(g)
    for i in comp_G.edges():
        q[i] = 2
    return q

def sort_2_arr(arr1, arr2):
    """
This function sorts two arrays (`arr1` and `arr2`) based on the values in `arr1` in descending order. 
The second array (`arr2`) is rearranged to maintain the correspondence with `arr1`.

Parameters:
- arr1: A list or array of numerical values used as the sorting key.
- arr2: A list or array of numerical values that is rearranged to match the order of `arr1`.

Returns:
- sort_1: A list containing the elements of `arr1` sorted in descending order.
- sort_2: A list containing the elements of `arr2` rearranged to match the sorted order of `arr1`.
"""
    sort_1 = []
    sort_2 = []
    buffer_1 = arr1.copy()
    buffer_2 = arr2.copy()
    for i in range(len(arr1)):
        id_1 = buffer_1.index(np.max(buffer_1))
        sort_1.append(buffer_1[id_1])
        sort_2.append(buffer_2[id_1])
        buffer_1.pop(id_1)
        buffer_2.pop(id_1)
    return sort_1, sort_2

def generate_qubo_value(g, func):
    """
    This function calculates a QUBO (Quadratic Unconstrained Binary Optimization) weight array 
    based on statistical differences between the attributes of nodes in the largest cliques 
    of a graph.

    Parameters:
    - g (networkx.Graph): The graph containing nodes with attributes 'w1', 'w2', 'w3', and 'w4'.
    - func (list): A list of cliques, where each element is a list containing nodes of the clique.

    Returns:
    - numpy.ndarray: An array of weights [w1, w2, w3, w4] assigned based on the 
      differences in mean attribute values between the two largest cliques.
      Value of the array are 1,2,4,8.
    """
    cliques = []
    for i in range(len(func)):
        cliques.append(func[i][0])
    
    # Ottieni gli attributi dei nodi
    w1_f = nx.get_node_attributes(g, "w1")
    w2_f = nx.get_node_attributes(g, "w2")
    w3_f = nx.get_node_attributes(g, "w3")
    w4_f = nx.get_node_attributes(g, "w4")
    
    # Calcola le medie per ciascun attributo e ciascuna cricca
    clique_stats = []
    for clique in cliques:
        w1_values = [w1_f[node] for node in clique if node in w1_f]
        w2_values = [w2_f[node] for node in clique if node in w2_f]
        w3_values = [w3_f[node] for node in clique if node in w3_f]
        w4_values = [w4_f[node] for node in clique if node in w4_f]
        
        clique_stats.append({
            'Len': len(clique),
            'Mean w1': np.mean(w1_values),
            'Mean w2': np.mean(w2_values),
            'Mean w3': np.mean(w3_values),
            'Mean w4': np.mean(w4_values)
        })
    
    # Ordina le cricche per dimensione e seleziona le due più grandi
    top_cliques = sorted(clique_stats, key=lambda x: x['Len'], reverse=True)[:2]
    
    # Calcola le differenze tra le medie delle due cricche principali
    differences = {
        'w1': top_cliques[1]['Mean w1'] - top_cliques[0]['Mean w1'],
        'w2': top_cliques[1]['Mean w2'] - top_cliques[0]['Mean w2'],
        'w3': top_cliques[1]['Mean w3'] - top_cliques[0]['Mean w3'],
        'w4': top_cliques[1]['Mean w4'] - top_cliques[0]['Mean w4'],
    }
    
    # Ordina le differenze in ordine decrescente
    sorted_diff = sorted(differences.items(), key=lambda x: x[1], reverse=True)
    
    # Assegna i pesi 8, 4, 2, 1
    weight_mapping = {sorted_diff[i][0]: weight for i, weight in enumerate([8, 4, 2, 1])}
    
    # Crea l'array dei pesi in ordine w1, w2, w3, w4
    result_array = np.array([weight_mapping['w1'], weight_mapping['w2'], weight_mapping['w3'], weight_mapping['w4']])
    
    return result_array