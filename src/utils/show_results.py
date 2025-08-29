import networkx as nx
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from .basic import func_max, sort_2_arr

def print_res(cli_list, g, num_print, return_max=False):
    """
    This function prints information about the top cliques in a graph, sorted by their objective function values. 

    Parameters:
    - cli_list: A list of cliques (each clique is represented as a list of nodes).
    - g: A NetworkX graph.
    - num_print: The number of top cliques to display.
    - return_max: return a list of cliques with maximum function value

    Returns:
    - max_cliques: if return_max = True, return a list of maximum cliques
    """
    values = [func_max(g, cli_list[i]) for i in range(len(cli_list))]
    index = [i for i in range(len(cli_list))]
    val, idx = sort_2_arr(values, index)
    if num_print <= len(val):
        times = num_print
    else:
        times = len(val)
    for i in range(times):
        print(f"{idx[i]}° cricca trovata: {cli_list[idx[i]]} \\\\")
        print(f"Valore funzione: {val[i]} \\\\")
        print("\\newline")
    if return_max:
        max_cliques = [cli_list[idx[j]] for j in range(len(val)) if val[j]==max(val)]
        return max_cliques



def print_func(cli_list, g, original, num_print):
    """
    This function prints information about the top cliques in a graph, sorted by their objective function values. 
    It compares the identified cliques to an original list of cliques and highlights matches.

    Parameters:
    - cli_list: A list of cliques (each clique is represented as a list of nodes).
    - g: A NetworkX graph.
    - original: A list of cliques considered as "original" or reference cliques.
    - num_print: The number of top cliques to display.
    """
    values = [func_max(g, cli_list[i]) for i in range(len(cli_list))]
    index = [i for i in range(len(cli_list))]
    val, idx = sort_2_arr(values, index)
    if num_print <= len(val):
        times = num_print
    else:
        times = len(val)
    for i in range(times):
        finded = False
        for j in original:
            if cli_list[idx[i]] == j:
                finded = True
        if finded:
            print(f"{idx[i]}° cricca trovata: {cli_list[idx[i]]} ed è una di quelle originali \\\\")
        else:
            print(f"{idx[i]}° cricca trovata: {cli_list[idx[i]]} \\\\")
        print(f"Valore funzione: {val[i]} \\\\")
        print("\\newline")

def extract_w(g, func):
    """
    Extracts node attributes (w1, w2, w3, w4) for cliques and calculates statistics.

    Parameters:
    - g (networkx.Graph): The input graph with node attributes.
    - func (list): A list of cliques, each represented as a list of nodes.

    Returns:
    - pandas.DataFrame: A DataFrame containing the mean and variance of each attribute
      (w1, w2, w3, w4) for each clique.
    """
    cliques = []
    for i in range(len(func)):
        cliques.append(func[i][0])
    w1_f = nx.get_node_attributes(g, "w1")
    w2_f = nx.get_node_attributes(g, "w2")
    w3_f = nx.get_node_attributes(g, "w3")
    w4_f = nx.get_node_attributes(g, "w4")
    # Calcola media e varianza per ciascun array in cliques
    data = []
    for clique in cliques:
    # Estrai i valori di w1_f, w2_f, w3_f, w4_f per le chiavi presenti in clique
        w1_values = [w1_f[node] for node in clique if node in w1_f]
        w2_values = [w2_f[node] for node in clique if node in w2_f]
        w3_values = [w3_f[node] for node in clique if node in w3_f]
        w4_values = [w4_f[node] for node in clique if node in w4_f]
        
        # Calcola media e varianza per ciascuna lista di valori
        mean_w1 = np.mean(w1_values)
        var_w1 = np.var(w1_values)
        
        mean_w2 = np.mean(w2_values)
        var_w2 = np.var(w2_values)
        
        mean_w3 = np.mean(w3_values)
        var_w3 = np.var(w3_values)
        
        mean_w4 = np.mean(w4_values)
        var_w4 = np.var(w4_values)
        
        data.append({
        'Clique': clique,
        'Len': len(clique),
        'Mean w1': round(mean_w1, 4),
        'Variance w1': round(var_w1, 4),
        'Mean w2': round(mean_w2, 4),
        'Variance w2': round(var_w2, 4),
        'Mean w3': round(mean_w3, 4),
        'Variance w3': round(var_w3, 4),
        'Mean w4': round(mean_w4, 4),
        'Variance w4': round(var_w4, 4)
    })

    # Creazione del DataFrame
    df = pd.DataFrame(data)
    return df

def extract_and_compare_max_means(df):
    """
    Extracts and compares the mean values of attributes for the two largest cliques 
    based on their size.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing clique statistics, including sizes 
      and mean values.

    Returns:
    - pandas.DataFrame: A DataFrame showing the comparison of means and differences 
      between the two largest cliques.
    """
    # Ordina il DataFrame in base alla colonna 'Len' in ordine decrescente e prendi le prime due righe
    top_rows = df.nlargest(2, 'Len')
    
    # Verifica se ci sono almeno due righe
    if top_rows.shape[0] < 2:
        raise ValueError("Il DataFrame deve contenere almeno due righe con valori differenti in 'Len'.")
    
    # Filtra le colonne che contengono solo le medie
    mean_columns = [col for col in df.columns if 'Mean' in col]
    
    # Prendi i valori delle medie per le due righe
    cricca1_values = top_rows.iloc[0][mean_columns].astype(float).values
    cricca2_values = top_rows.iloc[1][mean_columns].astype(float).values
    
    # Calcola le differenze
    differences = np.round(cricca1_values - cricca2_values, 5)
    
    # Crea il nuovo DataFrame
    diff_data = {
        'Metric': mean_columns,  # Nomi delle metriche (medie)
        'Cricca Massima': cricca1_values,
        'Cricca cercata': cricca2_values,
        'Difference': differences
    }
    
    diff_df = pd.DataFrame(diff_data)
    return diff_df

def extract_numbers(filename):
    """
    Extracts the leading numerical value from a folder.

    Parameters:
    - filename (str): The input folder.

    Returns:
    - int or float: The extracted numerical value, or infinity if no match is found.
    """
    match = re.match(r"(\d+)", filename)
    return int(match.group(0)) if match else float('inf')

def print_graph(dataframe, dataframe_diff, name, pos, fold):
    """
    Plots a graphical representation of clique frequency data, including a comparison 
    table for the two largest cliques.

    Parameters:
    - dataframe (pandas.DataFrame): DataFrame containing frequency data for cliques.
    - dataframe_diff (pandas.DataFrame): DataFrame with differences between the two largest cliques.
    - name (str): The name of the graph.
    - pos (int): Position for arranging tables within the plot.
    - fold (str): Folder path to save the output figure.

    Steps:
    1. Plot frequency data for each clique.
    2. Add tables summarizing the clique statistics and differences.
    3. Save the plot as a PNG file and display it.
    """
    plt.figure(figsize=(12, 8))
    for idx, row in dataframe.iterrows():
        # Escludo i valori dopo il primo -1
        if -1 in row.values:
            valid_values = row.iloc[:row.values.tolist().index(-1)]
        else:
            valid_values = row
        if valid_values.sum() != 0:
            last_non_zero_idx = valid_values[valid_values != 0].index[-1]
            final_values = valid_values.loc[:last_non_zero_idx]
        else: 
            final_values = valid_values
        plt.plot([idx for idx in final_values.index if final_values[idx]!=0] , final_values[final_values!=0], 
                label=f'Riga {idx}', 
                linestyle='-', 
                marker='o', 
                alpha=0.8, 
                color='red' if idx == 0 else None)  # Riga 0 in rosso

    plt.yscale('log', base=10)
    plt.xticks(ticks=dataframe.columns, rotation=45)  # Mostra tutti i valori dell'asse X con rotazione
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)  # Aggiunge la griglia
    plt.title(name)
    plt.xlabel('Dimensione Cricca')
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.set_ticks_position('top')
    plt.ylabel('Frequenza (Log)')
    plt.legend(title='Righe', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.table(cellText=dataframe.values, colLabels=dataframe.columns, loc='bottom', cellLoc='center', colColours=['lightgray']*dataframe.shape[1])
    plt.subplots_adjust(bottom=0.4)
    if pos == 0:
        table_bbox = [0, 0.8, 0.4, 0.15]
    else:
        table_bbox = [0.60, 0.8, 0.4, 0.15]
    table_diff = plt.table(
        cellText=dataframe_diff.values,
        colLabels=dataframe_diff.columns,
        loc='upper left',
        cellLoc='center',
        colColours=['lightblue'] * dataframe_diff.shape[1],
        bbox=table_bbox # [x, y, width, height] all'interno del grafico
    )

    # Riduzione della dimensione della seconda tabella
    table_diff.auto_set_font_size(False)
    table_diff.set_fontsize(8)
    plt.tight_layout()
    #save
    name_cut = re.sub(r'[<>:"/\\|?*]', '_', name.replace(".pkl", ""))
    output_path = f"{fold}/figures/{name_cut}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight') 
    plt.show()

def print_graph_nodf(dataframe, name):
    """
    Plots a graphical representation of clique frequency data without additional comparison tables.

    Parameters:
    - dataframe (pandas.DataFrame): DataFrame containing frequency data for cliques.
    - name (str): The name of the graph.

    Steps:
    1. Plot frequency data for each clique.
    2. Display the plot with frequency (log scale) and clique size.
    """
    plt.figure(figsize=(12, 8))
    for idx, row in dataframe.iterrows():
        # Escludo i valori dopo il primo -1
        if -1 in row.values:
            valid_values = row.iloc[:row.values.tolist().index(-1)]
        else:
            valid_values = row
        if valid_values.sum() != 0:
            last_non_zero_idx = valid_values[valid_values != 0].index[-1]
            final_values = valid_values.loc[:last_non_zero_idx]
        else: 
            final_values = valid_values
        plt.plot([idx for idx in final_values.index if final_values[idx]!=0] , final_values[final_values!=0], 
                label=f'Riga {idx}', 
                linestyle='-', 
                marker='o', 
                alpha=0.8, 
                color='red' if idx == 0 else None)  # Riga 0 in rosso

    plt.yscale('log', base=10)
    plt.xticks(ticks=dataframe.columns, rotation=45)  # Mostra tutti i valori dell'asse X con rotazione
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)  # Aggiunge la griglia
    plt.title(name)
    plt.xlabel('Dimensione Cricca')
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.set_ticks_position('top')
    plt.ylabel('Frequenza (Log)')
    plt.legend(title='Righe', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.table(cellText=dataframe.values, colLabels=dataframe.columns, loc='bottom', cellLoc='center', colColours=['lightgray']*dataframe.shape[1])
    plt.subplots_adjust(bottom=0.4)
    plt.tight_layout()
    plt.show()