import dwave_networkx as dnx
import pandas as pd
from .basic import QUBO_bias

def calculate_results(G, sampler, bias, boolean, reads, annel_time=None, init=None, anneal_schedule=None):
    """
    Calculates the maximum cliques in a graph using a quantum sampler.

    Parameters:
    - G (networkx.Graph): The input graph.
    - sampler: A quantum sampler object for QUBO problems.
    - bias: Boolean that indicate if a Bias is used for QUBO generation.
    - boolean: A Boolean parameter to adjust QUBO generation. True if the QUBO function must be weighted.
    - reads (int): Number of reads or samples to be taken.
    - annel_time (optional, int): The annealing time for the sampler.
    - init (optional, dict): Initial state for the sampler.
    - anneal_schedule (optional, list): Custom annealing schedule.

    Returns:
    - list: A list of maximum cliques found in the graph.

    Steps:
    1. Generate the QUBO matrix using the `QUBO_bias` function.
    2. Sample the QUBO using the provided sampler with optional parameters.
    3. Extract occurrences where nodes have positive values.
    4. Verify if the occurrences form cliques in the graph.
    5. Return all valid maximum cliques.
    """
    qubo = QUBO_bias(G, bias, boolean, [1,2,4,8])
    if init is not None and anneal_schedule is not None and annel_time is None :
        response = sampler.sample_qubo(qubo, num_reads=reads, initial_state=init, anneal_schedule=anneal_schedule)
    elif init is None and anneal_schedule is None and annel_time is not None:
        response = sampler.sample_qubo(qubo, num_reads=reads, annealing_time = annel_time)
    else:
        response = sampler.sample_qubo(qubo, num_reads=reads)
    occurrences = []
    for sample, num_occurrences in zip(response, response.record['num_occurrences']):
        data = [node for node in sample if sample[node] > 0]
        occurrences.extend([data] * num_occurrences)
    max_clique=[]
    for cli in occurrences:
        if dnx.is_clique(G, cli):
            max_clique.append(cli)
    return max_clique

def generate_empty_dataframe(func):
    """
    Generates an empty pandas DataFrame for storing intermediate solutions.
    Every row, except for the biggest clique, have a -1 that indicates the lenght of the clique.

    Parameters:
    - func (list): A list of cliques, where each clique is a list of nodes.

    Returns:
    - tuple:
        - pandas.DataFrame: An empty DataFrame initialized for clique data.
        - list: A list of cliques extracted from the input.
    """
    cliques = []
    name = []
    for i in range(len(func)):
        cliques.append(func[i][0])
        name.append(i)
    max_len = max(len(a) for a in cliques)
    solutions = pd.DataFrame(columns=[j for j in range(max_len+1)])
    start_values = [0 for _ in range(max_len+1)]
    for i in range(len(name)):
        solutions.loc[name[i]] = start_values
        current = len(cliques[i])
        if current != max_len:
            solutions.iloc[name[i], current+1] = -1
    return solutions, cliques

def find_intermediate_solutions(g, sampler, dataframe, cli_list, qubo, reads, init=None, anneal_schedule=None):
    """
    Updates a DataFrame with intermediate solutions found using the sampler.

    Parameters:
    - g (networkx.Graph): The input graph.
    - sampler: A quantum sampler object for QUBO problems.
    - dataframe (pandas.DataFrame): DataFrame to store intermediate results.
    - cli_list (list): List of cliques to be used for comparison.
    - qubo: QUBO matrix for sampling.
    - reads (int): Number of reads or samples to be taken.
    - init (optional, dict): Initial state for the sampler.
    - anneal_schedule (optional, list): Custom annealing schedule.

    Returns:
    - pandas.DataFrame: The updated DataFrame with intermediate results
    """
    max_clique = []
    if init is None and anneal_schedule is None:
        response = sampler.sample_qubo(qubo, num_reads=reads)
    else:
        response = sampler.sample_qubo(qubo, num_reads=reads, initial_state=init, anneal_schedule=anneal_schedule)
    occurrences = list([[node for node in occ if occ[node] > 0] for occ in response] )
    num_occurrences = list(response.record.num_occurrences)
    for idx,cli in enumerate(occurrences):
        if dnx.is_clique(g, cli):
            max_clique.append(cli)
            k = next((k for k in range(len(cli_list)) if all(el in cli_list[k] for el in cli)), None)
            if k is not None:
                dataframe.iloc[k,len(cli)] += num_occurrences[idx]
    return dataframe