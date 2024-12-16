import os
import numpy as np
from itertools import product
import pickle

# global CA parameters
width = 5 # width of the 1D cellular automaton
n_steps = 2**width # number of steps to simulate
initial_states = [np.array(bits) for bits in product([0, 1], repeat=width)] # all possible initial states
rules = [30, 45, 73, 110, 150, 105, 54, 22, 60, 146, 126, 62, 90, 18, 122, 26, 154, 94, 41, 57, 156, 28, 58, 78,
            178, 77, 50, 13, 25, 37, 9, 35, 106, 3, 27, 43, 184, 56, 11, 142, 14, 134, 24, 7, 152, 170, 46, 15, 33,
            1, 42, 162, 6, 5, 138, 38, 10, 74, 34, 29, 130, 2, 204, 200, 172, 108, 76, 72, 51, 44, 104, 232, 140, 132,
            23, 12, 164, 36, 19, 4, 168, 40, 160, 136, 128, 32, 8, 0]

# Powers of 2 up to the width
# Pre-computed quantity to speed up the conversion from
# binary representation of CA state to integer.
powers_of_2 = 2 ** np.arange(width)[::-1]

def rule_index(triplet):
    L, C, R = triplet
    index = 7 - (4 * L + 2 * C + R)
    return int(index)

def CA_run(initial_state, n_steps, rule_number):
    """
    Simulates a one-dimensional cellular automaton (CA) for a given number of steps.
    Parameters:
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

def CA_1D_State_To_Integer(state):
    """
    Converts a cellular automaton (CA) state to an integer.
    This function takes a list or array representing the state of a cellular automaton,
    where each element is a binary value (0 or 1), and converts it to an integer using
    a dot product with a predefined array of powers of 2.
    Args:
        state (list or np.ndarray): A list or array of binary values representing the CA state.
    Returns:
        int: The integer representation of the CA state.
    """
    return np.dot(state, powers_of_2)

def Inverse_CA_1D_State(integer_state):
    """
    Converts an integer to a binary representation of a cellular automaton (CA) state.
    This function takes an integer and converts it to a binary array representing the state
    of a cellular automaton (CA) of a given width.
    Args:
        integer_state (int): The integer representation of the CA state.
    Returns:
        np.ndarray: A binary array representing the CA state.
    """
    return np.array([int(x) for x in np.binary_repr(integer_state, int)])

def Check_Make_Dir(path):
    """
    Checks if a directory exists at the given path, and if not, creates it.
    Args:
        path (str): The path where the directory should be checked/created.
    Returns:
        None
    """
    if not os.path.exists(path):
        os.mkdir(path)

def Count_1D_Space_patterns(_width, _n_steps, _rules, _initial_states,
                            _in_trajectories_path=".", _in_trajectories_file_name = "trajectories_",
                            _out_patterns_1D_space_path=".", _out_patterns_1D_space_file_name = "all_pattern_counts_"):
    """
    Count the occurrences of 1D Cellular Automata (CA) patterns over a number of steps and save the results.
    Parameters:
    _width (int): The width of the 1D CA.
    _n_steps (int): The number of steps to simulate the CA.
    _rules (list): The list of rules (as integers) for which the trajectories were recorded.
    _initial_states (list): The list of initial states for which the trajectories were recorded.
    _in_trajectories_path (str): The path to the directory containing the trajectory for the CA of the given width.
    _in_trajectories_file_name (str): The base name of the file containing the trajectories for the CA of the given width.
    _out_patterns_1D_space_path (str): The path to the directory where the pattern count results will be saved.
    _out_patterns_1D_space_file_name (str): The base name of the file where the pattern count results will be saved.
    Returns:
    None
    """

    # Extract the trajectory from the recorded data
    file = open(_in_trajectories_path + "/" + _in_trajectories_file_name + str(_width), 'rb')
    trajectories = pickle.load(file) # dict with double keys: rule and initial state
    file.close()

    # Dictionary to store the patterns. Keys are the rule number and the initial state
    all_pattern_counts = {_rule: {} for _rule in _rules}
    # Make an array full of zeros to store the number of times each CA state appears
    pattern_counts = np.zeros(2**_width, dtype=int)

    # Loop over all rules and initial states
    counter = 0 # to keep track of the progress
    nb_rules = len(_rules)
    for _rule in _rules:
        counter += 1
        print(f"Rule {_rule} ({counter}/{nb_rules})", end="\r")
        pattern_counts *= 0 # reset the pattern counts
        for initial_state in _initial_states:
            CA = trajectories[_rule][str(initial_state)]
            # Analyze the CA to see if it matches any of the predefined patterns
            # note: the patterns are converted to their integer representation
            for i in range(_n_steps):
                pattern_counts[CA_1D_State_To_Integer(CA[i, :])] += 1
            all_pattern_counts[_rule][str(initial_state)] = pattern_counts
            
    # save the pattern in a byte file
    filename = _out_patterns_1D_space_path + "/" + _out_patterns_1D_space_file_name + str(_width)
    file = open(filename, 'wb')
    pickle.dump(all_pattern_counts, file)
    file.close()

if __name__ == "__main__":

    #paths for the data that will be generated
    root = "./data"
    patterns_path = root + "/patterns"
    trajectories_path = root + "/trajectories"
    pinned_trajectories_path = root + "/pinned_trajectories"
    patterns_1D_path = root + "/patterns/1D"
    patterns_1D_space_path = root + "/patterns/1D/space" # for patterns along the 1D space at a single time step
    patterns_1D_time_path = root + "/patterns/1D/time" # for patterns along the time axis at a single space cell

    # Check the directories and create them if they don't exist
    for path in [root, patterns_path, patterns_1D_path, patterns_1D_space_path, patterns_1D_time_path]:
        Check_Make_Dir(path)

    # Count the patterns along the 1D space
    # Count_1D_Space_patterns(width, n_steps, rules, initial_states, trajectories_path, patterns_1D_space_path)

    # for checking purposes, open the file and print the patterns
    file = open(patterns_1D_space_path + f"/all_pattern_counts_{width}", 'rb')
    all_pattern_counts = pickle.load(file)
    file.close()
    print("Printing from the file read. The pattern counts for rule 30, initial state 00001 are:\n",
          all_pattern_counts[30][str(np.array([0, 0, 0, 0, 1]))])
    

