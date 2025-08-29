import math
import numpy as np
from tqdm import tqdm
from .basic import func_max

def is_optimal_solution(g, sample, opt):
    """
    Checks whether a given sample represents the optimal solution.

    Parameters:
    - g (networkx.Graph): The graph containing the problem structure.
    - sample (dict): A sample solution, with nodes as keys and states (0 or 1) as values.
    - opt (list): The optimal solution represented as a list of nodes.

    Returns:
    - bool: True if the sample matches the optimal solution, False otherwise.
    """
    sol = [nodo for nodo, stato in sample.items() if stato == 1]
    return func_max(g, sol) == func_max(g, opt)

def calculate_repetitions(p_s, target_probability):
    """
    Calculates the number of repetitions needed to achieve the target success probability.

    Parameters:
    - p_s (float): The probability of success for a single attempt.
    - target_probability (float): The desired overall success probability.

    Returns:
    - int or float: The number of repetitions needed, or infinity if p_s is 0.
    """
    if p_s == 1:
        return 1
    elif p_s == 0:
        return float('inf')
    else:
        return math.ceil(math.log(1 - target_probability) / math.log(1 - p_s))

def compute_tts(qubo, g, sampler, opt, annealing_times, target_probability, num_repetition):
    """
    Computes the Total Time to Solution (TTS) for a quantum annealer over various annealing times.

    Parameters:
    - qubo (dict): The QUBO problem to be solved.
    - g (networkx.Graph): The graph containing the problem structure.
    - sampler: A quantum annealer sampler object.
    - opt (list): The optimal solution represented as a list of nodes.
    - annealing_times (list): A list of annealing times to test.
    - target_probability (float): The desired overall success probability.
    - num_repetition (int): The number of repetitions per annealing time.

    Returns:
    - dict: A dictionary mapping annealing times to tuples of (mean TTS, standard deviation TTS).

    Steps:
    1. For each annealing time, perform multiple sampling repetitions.
    2. Calculate the success probability for each repetition.
    3. Compute the TTS using the success probability and annealing time.
    4. Aggregate the TTS values and compute mean and standard deviation.
    5. Identify the optimal annealing time based on the lowest TTS.
    """
    tts_values_aggregated = {t_f: [] for t_f in annealing_times}
    for t_f in tqdm(annealing_times):
        for i in range(num_repetition):
            sampleset = sampler.sample_qubo(qubo, num_reads=2000, annealing_time=t_f)
            success_count = sum(1 for sample in sampleset if is_optimal_solution(g, sample, opt))
            p_s = success_count / 2000
            print(f"La prob di successo di {t_f} alla {i}° iterazione è: {p_s}")
            if p_s >0:
                R_t_f = calculate_repetitions(p_s, target_probability)
                tts = t_f * R_t_f
                tts_values_aggregated[t_f].append(tts)      

    # Media e std finale del TTS su tutte le ripetizioni
    tts_final = {}
    for t_f, tts_values in tts_values_aggregated.items():
        if tts_values:  # Considera solo se ci sono valori finiti
            tts_final[t_f] = (np.mean(tts_values), np.std(tts_values))
        else:
            # Se tutte le ripetizioni hanno prodotto inf, assegna un valore simbolico per l'invalidità
            tts_final[t_f] = (float("inf"), float("inf"))
    
    # Identifica il TTS ottimale tra i valori finiti
    finite_tts = {t_f: stats for t_f, stats in tts_final.items() if stats[0] != float("inf")}
    if finite_tts:
        optimal_t_f = min(finite_tts, key=lambda x: finite_tts[x][0])
        optimal_tts, optimal_std = finite_tts[optimal_t_f]
        print(f"Optimal annealing time: {optimal_t_f} µs with TTS: {optimal_tts} µs ± {optimal_std} µs")
    else:
        print("No valid TTS found with the given annealing times and repeats.")
    return tts_final