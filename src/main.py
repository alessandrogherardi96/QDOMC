from utils.graph_creation import *
from utils.weights_graph import *
import json
import pickle
import os
import pandas as pd

def main():
    a=0
    b=0.027290137282490516
    c=d=0.03640364796375065
    e=f=0.08197120137005134
    const = [a,b,c,d,e,f]
    while True:
        try:
            number_of_graphs = int(input("How many graph do you want to create?"))
            if number_of_graphs > 0:
                break  
            else:
                print("Number must be > 0")
        except ValueError:
            print("Not an integer")
    print("What function do you want to use? \n 1: graph_with_clique_density, use a random number of iteration based on denity \n 2: graph_with_clique_degree, use degree values for creating edges")
    while True:
        try:
            func = int(input())
            if func == 1 or func== 2:
                break  
            else:
                print("Number must be 1 or 2")
        except ValueError:
            print("Not an integer")
    if func == 1:
        print( "Insert list of values separated by spaces \n 1- num_node integer: number of nodes per clique \n 2- ex_node integer: number of extra nodes outside the cliques")
        print(" 3- num_clique integer: number of cliques in the graph (>0) \n 4- density integer: density of the graph, used to define edge connections (>0)")
        print(" 5- add_edges_between_cliques boolean (0, 1): if True, adds edges between cliques \n 6- random_clique_dimension boolean (0, 1): whether to randomize the size of cliques")
        while True:
            try:
                num_node, ex_node, num_clique, density, add_edges_between_cliques, random_clique_dimension = input().split()
                num_node = int(num_node)
                ex_node = int(ex_node)
                num_clique = int(num_clique)
                density = int(density)
                add_edges_between_cliques = bool(int(add_edges_between_cliques))
                random_clique_dimension = bool(int(random_clique_dimension))
                if (num_node > 0 and ex_node >0 and num_clique >0 and density>0):
                    break
                else:
                    print("Something insert is wrong1")
            except ValueError:
                print("Missing values or something insert is wrong")
    else:
        print( "Insert list of values separated by spaces \n 1- num_node integer: number of nodes per clique \n 2- ex_node integer: number of extra nodes outside the cliques")
        print(" 3- num_clique integer: number of cliques in the graph (>0) \n 4- degree_min float (between 0 and 1, not 0): minimum value of the ratio node_degree/total_nodes")
        print(" 5- degree_max float (between 0 and 1, not 0): maximum value of the ratio node_degree/total_nodes")
        print(" 5- add_edges_between_cliques boolean (0, 1): if True, adds edges between cliques \n 6- random_clique_dimension boolean (0, 1): whether to randomize the size of cliques")
        while True:
            try:
                num_node, ex_node, num_clique, degree_min, degree_max, add_edges_between_cliques, random_clique_dimension = input().split()
                num_node = int(num_node)
                ex_node = int(ex_node)
                num_clique = int(num_clique)
                degree_min = float(degree_min)
                degree_max = float(degree_max)
                add_edges_between_cliques = bool(int(add_edges_between_cliques))
                random_clique_dimension = bool(int(random_clique_dimension))
                if (num_node > 0 and ex_node >0 and num_clique >0 and 
                    degree_max>0 and degree_max<=1 and degree_min>0 and degree_min<=1
                    and degree_max>degree_min):
                    break
                else:
                    print("Something insert is wrong")
            except ValueError:
                print("Missing values or something insert is wrong")
    print("Importing probability matrices...")
    with open('./src/probability_matrices/matrix_w00.pkl', 'rb') as f:
        matrice_w00 = pickle.load(f)
    with open('./src/probability_matrices/matrix_w01.pkl', 'rb') as f:
        matrice_w01 = pickle.load(f)
    for i in range(number_of_graphs):
        print(f"Creating {i+1}° graph \n")
        if func == 1:
            g, cli = graph_with_clique_density(num_node, ex_node, num_clique, density, add_edges_between_cliques, random_clique_dimension)
            print("")
        else:
            g, cli = graph_with_clique_degree(num_node, ex_node, num_clique, degree_min, degree_max, add_edges_between_cliques, random_clique_dimension)
            print("")
        print("Calculating random weights for the graph...")
        dict_G = assign_random_weights(g, matrice_w00, matrice_w01)
        set_graph_weights(g, dict_G, const)
        print("Saving graph to output...\n")
        folder_g = f"./output/graphs/{num_node}_{ex_node}_{num_clique}"
        folder_cli = f"./output/cliques/{num_node}_{ex_node}_{num_clique}"
        if not os.path.exists(folder_g):
            os.makedirs(folder_g)
        if not os.path.exists(folder_cli):
            os.makedirs(folder_cli)
        k = 0
        while True:
            path_g = f"./output/graphs/{num_node}_{ex_node}_{num_clique}/{k}.pkl"
            path_cli = f"./output/cliques/{num_node}_{ex_node}_{num_clique}/{k}.pkl"
            if not os.path.exists(path_g) and not os.path.exists(path_cli):
                break
            else:
                k += 1
        with open(path_g, "wb") as f:
            pickle.dump(g, f)
        with open(path_cli, "wb") as f:
            pickle.dump(cli, f)
        

if __name__ == "__main__":
    main()