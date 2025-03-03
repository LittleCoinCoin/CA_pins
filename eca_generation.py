import pickle
import numpy as np
from itertools import product
from concurrent.futures import ProcessPoolExecutor

def rule_index(triplet):
    L, C, R = triplet
    index = 7 - (4 * L + 2 * C + R)
    return int(index)

def CA_run(initial_state, n_steps, rule_number):
    """
    Simulates a one-dimensional cellular automaton (CA) for a given number of steps.
    Args:
        initial_state (list or array-like): The initial state of the CA, a list or array of integers (0 or 1).
        n_steps (int): The number of steps to simulate.
        rule_number (int): The rule number (0-255) that defines the CA's behavior, based on Wolfram's rule numbering.
    Returns:
        numpy.ndarray: A 2D array where each row represents the state of the CA at a given step.
    """

    rule_string = np.binary_repr(rule_number, 8)
    rule = np.array([int(bit) for bit in rule_string])

    m_cells = len(initial_state)
    CA_run = np.zeros((n_steps, m_cells), dtype=int)
    CA_run[0, :] = initial_state

    for step in range(1, n_steps):
        all_triplets = np.stack(
            [
                np.roll(CA_run[step - 1, :], 1),
                CA_run[step - 1, :],
                np.roll(CA_run[step - 1, :], -1),
            ]
        )
        CA_run[step, :] = rule[np.apply_along_axis(rule_index, 0, all_triplets)]

    return CA_run

def process_rule(_width, _rule, _ICs):
    """
    Processes a given CA rule and initial conditions (ICs) to generate rule trajectories.
    Args:
        _width (int): The width of the 1D CA. It MUST match the width of the ICs.
        _rule (int): The rule number to be processed.
        _ICs (list): A list of initial conditions to be used for generating trajectories. They MUST
                     match the ICs
    Returns:
        tuple: A tuple containing the rule and a dictionary of rule trajectories.
               The dictionary keys are string representations of initial conditions,
               and the values are the results of the CA_run function.
    """
    rule_trajectories = {}
    for ic in _ICs:
        rule_trajectories[str(ic)] = CA_run(ic, 2**_width, _rule)
    return _rule, rule_trajectories

def eca_trajectories(_width, _rules, _ICs, _path=".", _out_file_name="trajectories", _num_workers = 4):
    """
    Generate and save trajectories for Elementary Cellular Automata (ECA) rules for a set of
    initial conditions (ICs)
    The function generates trajectories for a set of ECA rules using parallel processing and saves
    the results as a pickle file in the specified directory.
    Args:
        _width (int): The width of the cellular automaton.
        _rules (list): A list of ECA rule numbers to generate trajectories for.
        _ICs (list or array-like): A collections of lists or array-like datastructures containing the array-like of
                                    binaries (0s or 1s) defining initial conditions of the ECA.
        _path (str, optional): The directory path where the output file will be saved. Default is the current directory.
        _out_file_name (str, optional): The base name of the output file. Default is "trajectories".
        _num_workers (int, optional): The number of worker processes to use for parallel processing. Default is 4.
    Returns:
        None
    
    """

    trajectories = {}

    with ProcessPoolExecutor(max_workers=_num_workers) as executor:
        futures = [executor.submit(process_rule,_width, r, _ICs) for r in _rules]
        for future in futures:
            rule, rule_trajectories = future.result()
            trajectories[rule] = rule_trajectories

    # save
    file = open(_path + "/" + _out_file_name + f"_{_width}", 'wb')
    pickle.dump(trajectories, file)
    file.close()


if __name__ == "__main__":
    #paths for the data that will be generated
    root = "./data"
    patterns_path = root + "/patterns"
    trajectories_path = root + "/trajectories"
    #pinned_trajectories_path = root + "/pinned_trajectories"

    # global CA parameters
    width = 5 # width of the 1D cellular automaton
    n_steps = 2**width # number of steps to simulate
    ICs = [np.array(bits) for bits in product([0, 1], repeat=width)] # all possible initial states
    rules = [30, 45, 73, 110, 150, 105, 54, 22, 60, 146, 126, 62, 90, 18, 122, 26, 154, 94, 41, 57, 156, 28, 58, 78,
                178, 77, 50, 13, 25, 37, 9, 35, 106, 3, 27, 43, 184, 56, 11, 142, 14, 134, 24, 7, 152, 170, 46, 15, 33,
                1, 42, 162, 6, 5, 138, 38, 10, 74, 34, 29, 130, 2, 204, 200, 172, 108, 76, 72, 51, 44, 104, 232, 140, 132,
                23, 12, 164, 36, 19, 4, 168, 40, 160, 136, 128, 32, 8, 0]

    # Generate the trajectories for the ECA rules for the corresponding width, ICs and rules
    eca_trajectories(width, rules, ICs, trajectories_path, "trajectories")