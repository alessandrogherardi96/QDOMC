import numpy as np
import pandas  as pd
import networkx as nx

def calculate_weights(k, value, cs):
    """
    Input:
        k (int): The key representing a node.
        value (list): A list containing two values associated with the node.
        cs (list): A list of constants used to claculate the weights.
    
    Output:
        dict: A dictionary with the node `k` as the key, and a list of 4 computed weights as the value.
    
    Explanation:
        This function calculates four different weights for a node based on its associated values 
        and the given constants.
    """
    v, w00, w01 = [k,value[0], value[1]]
    w1 = float(min(max(w00-cs[0], 0), cs[1]))
    w2 = float(min(max(w00-cs[2], 0), cs[3]))
    w3 = float(min(max((w01-cs[4])*0.5,0), cs[5]*0.5))
    w4 = w3
    return {v:[w1,w2,w3,w4]}

def create_weights_dict(graph_dict, cs):
    """
    Input:
        graph_dict (dict): A dictionary where each key represents a node, and the value is a list of values 
                           associated with that node.
        cs (list): A list of constants to be used when calculating weights.
    
    Output:
        dict: A dictionary where each key is a node, and the value is a list of four weights calculated 
              using the `calculate_weights` function.
    
    Explanation:
        This function iterates over each node in the `graph_dict`, calculates its weights using 
        `calculate_weights`, and updates the overall weights dictionary.
    """
    weights = dict()
    for key in graph_dict:
        weights_values = calculate_weights(key, graph_dict[key], cs)
        weights.update(weights_values)
    return weights

def set_graph_weights(g, weights_dict, cs):
    """
    Input:
        g (networkx.Graph): A NetworkX graph object.
        weights_dict (dict): A dictionary with node keys and their respective weight values.
        cs (list): A list of constants used in weight calculation.
    
    Output:
        None
    
    Explanation:
        This function takes a graph and assigns the computed weights to the nodes as attributes 
        ('w1', 'w2', 'w3', 'w4'). It retrieves the weights for each node from `weights_dict` and 
        updates the graph with those values.
    """
    graph_weights = create_weights_dict(weights_dict, cs)
    nx.set_node_attributes(g, {int(k):v[0] for k,v in graph_weights.items()}, 'w1')
    nx.set_node_attributes(g, {int(k):v[1] for k,v in graph_weights.items()}, 'w2')
    nx.set_node_attributes(g, {int(k):v[2] for k,v in graph_weights.items()}, 'w3')
    nx.set_node_attributes(g, {int(k):v[3] for k,v in graph_weights.items()}, 'w4')

def create_probability_matrix(ratios, weights):
    """
    Input:
        ratios (list): List of ratios.
        weights (list): List of weights associated with the ratios.
    
    Output:
        pd.DataFrame: A DataFrame representing the probability matrix, where each cell contains 
                      the probability of a particular ratio being associated with a specific weight.

    Explanation:
        This function takes two lists: `ratios` and `weights`. It first identifies the unique values 
        from both lists and creates an empty matrix (`probability_matrix`) with the dimensions determined 
        by the unique counts of ratios and weights. It then calculates how many times each ratio is 
        associated with each weight by iterating over unique combinations of `ratios` and `weights`, 
        and stores the counts in the matrix.

        The counts are then normalized row-wise (i.e., per ratio), turning the counts into probabilities.
        Finally, the function returns a DataFrame (`probability_matrix_df`) with the unique ratios as index,
        unique weights as columns, and the computed probabilities as values.
    """
    # Get the unique values of ratios and weights
    unique_ratios = sorted(set(ratios))
    unique_weights = sorted(set(weights))

    # Create an empty matrix for the probabilities
    probability_matrix = np.zeros((len(unique_ratios), len(unique_weights)))

    # Create a DataFrame for easier handling
    df = pd.DataFrame({'ratio': ratios, 'weight': weights})

    # Calculate the probability for each ratio-weight combination
    for i, ratio in enumerate(unique_ratios):
        for j, weight in enumerate(unique_weights):
            # Count how many times the ratio is associated with that weight
            count = len(df[(df['ratio'] == ratio) & (df['weight'] == weight)])
            # Calculate the probability and insert it into the matrix
            probability_matrix[i, j] = count

    # Create a DataFrame to display the matrix
    probability_matrix_df = pd.DataFrame(probability_matrix, index=unique_ratios, columns=unique_weights)

    row_sums = probability_matrix_df.sum(axis=1)  # Sum of each row
    probability_matrix_df = probability_matrix_df.div(row_sums, axis=0)

    return probability_matrix_df

def generate_probability_matrix(g, graph_dict, save=False, pathw0="", pathw1=""):
    """
    Input:
        g (networkx.Graph): A NetworkX graph object representing the graph.
        graph_dict (dict): A dictionary where each key is a node, and the value is a list of two weights [w00, w01].
        save (bool): A flag indicating whether to save the resulting matrices to CSV files. Default is False.
        pathw0 (str): The file path for saving the w00 matrix if `save` is True.
        pathw1 (str): The file path for saving the w01 matrix if `save` is True.

    Output:
        tuple: A tuple containing two DataFrames:
            - matrix_w00: A DataFrame representing the probability matrix for weights w00.
            - matrix_w01: A DataFrame representing the probability matrix for weights w01.

    Explanation:
        This function generates two probability matrices for the weights `w00` and `w01` from the `graph_dict`.
        First, it extracts node names, their corresponding weights `w00` and `w01`, and computes the degree of 
        each node relative to the total number of nodes in the graph. The degree and weight lists are rounded 
        for precision. These values are then passed to the `create_probability_matrix` function to create 
        probability matrices for `w00` and `w01`. If `save` is set to True, the matrices are saved to CSV files 
        at the specified paths. The function returns the two matrices.
    """
    name = [int(key) for key in graph_dict]
    w00 = [graph_dict[key][0] for key in graph_dict]
    w01 = [graph_dict[key][1] for key in graph_dict]
    degrees = [g.degree()[k]/len(g) for k in name]
    degrees = np.round(degrees, 3)
    w00 = np.round(w00, 4)
    w01 = np.round(w01, 4)
    matrix_w00 = create_probability_matrix(degrees, w00)
    matrix_w01 = create_probability_matrix(degrees, w01)
    if save:
        matrix_w00.to_csv(pathw0, index=True)
        matrix_w01.to_csv(pathw1, index=True)
    return matrix_w00, matrix_w01

def assign_random_weights(graph, df0, df1):
    """
    Input:
        graph (networkx.Graph): A NetworkX graph object representing the structure with nodes.
        df0 (pd.DataFrame): A DataFrame containing probabilities and weights for w00, with ratios as the index.
        df1 (pd.DataFrame): A DataFrame containing probabilities and weights for w01, with ratios as the index.

    Output:
        dict: A dictionary where the keys are the graph nodes and the values are lists of two weights [w00_weight, w01_weight]
              assigned randomly according to the probability distributions.

    Explanation:
        This function assigns random weights to each node in a graph using probability distributions from the 
        DataFrames `df0` and `df1`. First, it calculates the ratio of the node's degree to the total number 
        of nodes in the graph. Then, for each node, it finds the row in the DataFrame with the closest ratio 
        and uses the probabilities in that row to randomly choose a weight for w00 and w01. If the chosen 
        weight is 0, it is replaced with a minimum value (0.00005). The result is a dictionary mapping each 
        node to its randomly assigned weights.
    """
    # Calculate the total number of nodes in the graph
    num_nodes = graph.number_of_nodes()
    node_weights = {}
    
    # Iterate over each node in the graph
    for node in graph.nodes():
        # Calculate the degree/number of nodes ratio for the current node
        degree = graph.degree[node]
        ratio = round(degree / num_nodes, 3)
        
        # Find the row in the w00 DataFrame corresponding to the closest ratio
        if ratio in df0.index:
            row0 = df0.loc[ratio]
        else:
            closest_ratio_idx0 = pd.DataFrame(df0.index - ratio).abs().idxmin()
            closest_ratio0 = df0.index[closest_ratio_idx0]
            row0 = df0.loc[closest_ratio0[0]]
        
        # Find the row in the w01 DataFrame corresponding to the closest ratio
        if ratio in df1.index:
            row1 = df1.loc[ratio]
        else:
            closest_ratio_idx1 = pd.DataFrame(df1.index - ratio).abs().idxmin()
            closest_ratio1 = df1.index[closest_ratio_idx1]
            row1 = df1.loc[closest_ratio1[0]]
        
        # Extract the probabilities from the row and normalize if necessary
        probability0 = row0.values
        weights0 = row0.index
        probability1 = row1.values
        weights1 = row1.index
        
        # Randomly choose a weight according to the probability distribution
        chosen_weight0 = np.random.choice(weights0, p=probability0)
        chosen_weight1 = np.random.choice(weights1, p=probability1)

        # Modify weights 0 if needed
        if chosen_weight0 == 0:
            chosen_weight0 = 0.00005
        if chosen_weight1 == 0:
            chosen_weight1 = 0.00005
        
        # Add the node and weight to the dictionary
        node_weights[node] = [chosen_weight0, chosen_weight1]
    
    return node_weights
