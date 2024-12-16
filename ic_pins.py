import os
import pickle
import numpy as np
from itertools import product
import copy
import seaborn as sns
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# global parameters
rules = [30, 45, 73, 110, 150, 105, 54, 22, 60, 146, 126, 62, 90, 18, 122, 26, 154, 94, 41, 57, 156, 28, 58, 78,
             178, 77, 50, 13, 25, 37, 9, 35, 106, 3, 27, 43, 184, 56, 11, 142, 14, 134, 24, 7, 152, 170, 46, 15, 33,
             1, 42, 162, 6, 5, 138, 38, 10, 74, 34, 29, 130, 2, 204, 200, 172, 108, 76, 72, 51, 44, 104, 232, 140, 132,
             23, 12, 164, 36, 19, 4, 168, 40, 160, 136, 128, 32, 8, 0]

def rule_index(triplet):

    L, C, R = triplet
    index = 7 - (4 * L + 2 * C + R)
    return int(index)

def CA_run(initial_state, n_steps, rule_number):

    rule_string = np.binary_repr(rule_number, 8)
    rule = np.array([int(bit) for bit in rule_string])

    m_cells = len(initial_state)
    CA_run = np.zeros((n_steps, m_cells))
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

    return CA_run.astype(int)

def process_rule(rule, _nb_steps, _initial_states):

        print("[eca_trajectories] Processing rule:", rule)

        trajectories_per_rule = {}

        for initial_state in _initial_states:
            trajectories_per_rule[str(initial_state)] = CA_run(initial_state, _nb_steps, rule)

        return rule, trajectories_per_rule

def eca_trajectories(width, _max_workers=4,
                     _out_trajectories_path=".",
                     _out_trajectories_file_name = "trajectories_"):

    initial_states = [np.array(bits) for bits in product([0, 1], repeat=width)]
    nb_steps = 2**width

    trajectories = {}

    with ProcessPoolExecutor(max_workers=_max_workers) as executor:
        results = [executor.submit(process_rule, rule, nb_steps, initial_states) for rule in rules]

    for _res in results:
        rule, trajectories_per_rule = _res.result()
        trajectories[rule] = trajectories_per_rule

    # save
    full_path = _out_trajectories_path+"/"+_out_trajectories_file_name + str(width)
    print("[eca_trajectories] Saving trajectories to: ", full_path)
    file = open(full_path, 'wb')
    pickle.dump(trajectories, file)
    file.close()


def process_pinned_rule(rule, width, initial_states, pins):

    print("[pinned_trajectories] Processing rule:", rule)

    trajectories_per_rule = {}
    for pin in pins:
        trajectories_per_rule[str(pin)] = {}
        for initial_state in initial_states:
            trajectories_per_rule[str(pin)][str(initial_state)] = {}
            initial_state_pinned = copy.deepcopy(initial_state)
            initial_state_pinned[pin[0][0]] = pin[1][0]
            initial_state_pinned[pin[0][1]] = pin[1][1]
            trajectories_per_rule[str(pin)][str(initial_state)][0] = initial_state_pinned
            t0 = initial_state_pinned
            for t in range(1, 2 ** width):
                data = CA_run(t0, 2, rule)
                t1 = data[-1]
                t1[pin[0][0]] = pin[1][0]
                t1[pin[0][1]] = pin[1][1]
                trajectories_per_rule[str(pin)][str(initial_state)][t] = t1
                t0 = t1
    return rule, trajectories_per_rule

def pinned_trajectories(width, _max_workers=4,
                        _out_pinned_trajectories_path=".",
                        _out_pinned_trajectories_file_name = "pinned_trajectories_"):
    trajectories = {}
    initial_states = [np.array(bits) for bits in product([0, 1], repeat=width)]
    pins = list(product(product(range(0, width), range(0, width)), product(range(0, 2), range(0, 2))))

    with ProcessPoolExecutor(max_workers=_max_workers) as executor:
        results = [executor.submit(process_pinned_rule, rule, width, initial_states, pins) for rule in rules]

    for _res in results:
        rule, trajectories_per_rule = _res.result()
        trajectories[rule] = trajectories_per_rule

    # save
    full_path = _out_pinned_trajectories_path+"/"+_out_pinned_trajectories_file_name+str(width)
    print("[pinned_trajectories] Saving pinned trajectories to: ", full_path)
    file = open(full_path, 'wb')
    pickle.dump(trajectories, file)
    file.close()


def calculate_results(width,
                      _in_trajectories_path=".", _in_trajectories_file_name = "trajectories_",
                      _in_pinned_trajectories_path=".", _in_pinned_trajectories_file_name = "pinned_trajectories_",
                      _out_results_path = ".", _out_results_file_name="pinned_results_"):

    class1 = [0, 8, 32, 40, 128, 136, 160, 168]

    class2 = [  1,  2,  3,  4,  5,  6,  7,      9,  10,
                11, 12, 13, 14, 15,             19,
                        23, 24, 25, 26, 27, 28, 29,
                        33, 34, 35, 36, 37, 38,
                    42, 43, 44,     46,
          50,   51,                 56, 57, 58,
                    62,
                    72, 73, 74,     76, 77, 78,
                            94,
                            104,            108,
         130,       132,    134,            138,
         140,       142,
                    152,    154,    156,
                    162,    164,
         170,       172,                    178,
                            184,
              200,          204,
                    232]

    class3 = [18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150]

    class4 = [41, 54, 106, 110]

    # regular ECA state space over all the rules
    full_path = _in_pinned_trajectories_path + "/" + _in_pinned_trajectories_file_name + str(width)
    if(not os.path.exists(full_path)):
        print("[calculate_results] Pinned trajectories not found at '" + full_path + "'.\nGenerating now.")
        pinned_trajectories(width,
                            _out_pinned_trajectories_path=_in_pinned_trajectories_path,
                            _out_pinned_trajectories_file_name = _in_pinned_trajectories_file_name)
    file = open(full_path, 'rb')
    pinned_trajectories = pickle.load(file)
    file.close()

    # all ECA trajectories up to 2^w
    full_path = _in_trajectories_path+"/"+_in_trajectories_file_name+ str(width - 2)
    if(not os.path.exists(full_path)):
        print("[calculate_results] Trajectories not found at '" + full_path + "'.\nGenerating now.")
        eca_trajectories(width - 2,
                         _out_trajectories_path = _in_trajectories_path,
                         _out_trajectories_file_name = _in_pinned_trajectories_file_name)
    file = open(full_path, 'rb')
    trajectories = pickle.load(file)
    file.close()

    initial_states = [np.array(bits) for bits in product([0, 1], repeat=width)]

    pins = list(product(product(range(0, width), range(0, width)), product(range(0, 2), range(0, 2))))  # (position, value)

    results = {}
    row = 0

    # loops over all rules
    for rule in rules:

        # loop over all single pins
        for pin in pins:

            # skip pins at the same site
            if pin[0][0] == pin[0][1]:
                continue

            # loop over all initial states
            for initial_state in initial_states:

                initial_state_label = str(initial_state)
                pinned_trajectory = pinned_trajectories[rule][str(pin)][initial_state_label]
                pinned_trajectory = np.array(list(pinned_trajectory.values()))

                # get C(r1)
                if rule in class1:
                    rule_c = "Class 1"
                elif rule in class2:
                    rule_c = "Class 2"
                elif rule in class3:
                    rule_c = "Class 3"
                else:
                    rule_c = "Class 4"

                # look at unpinned ECA trajectory
                unpinned_trajectory = np.delete(np.vstack(pinned_trajectory), pin[0], 1)

                # is it an ECA trajectory? If yes, then keep. if no then skip
                rule2 = None
                rule2s = []
                for rule2_check in rules:
                    smol_traj = trajectories[rule2_check][str(unpinned_trajectory[0])]
                    if str(smol_traj) == str(unpinned_trajectory[:2**(width-len(pin[0]))]):

                        # which ECA rule(s) is it?
                        rule2 = rule2_check
                        # add to list of potential rules
                        rule2s.append(rule2)

                    else:
                        continue
                if not rule2:
                    continue

                rule2_cs = []

                for rule2 in rule2s:

                    # get C(r2)
                    if rule2 in class1:
                        rule2_c = "Class 1"
                    elif rule2 in class2:
                        rule2_c = "Class 2"
                    elif rule2 in class3:
                        rule2_c = "Class 3"
                    else:
                        rule2_c = "Class 4"

                    rule2_cs.append(rule2_c)

                # get max C(r2)
                rule2_c = sorted(rule2_cs)[-1]

                # get the number of rules it could be
                n_rules2 = len(rule2s)

                # n_states
                n_states = len(set([str(unpinned_trajectory[x]) for x in range(0, len(unpinned_trajectory))]))
                if n_states > 2:
                    print([rule, rule_c, initial_state, pin[0], pin[1], rule2, rule2_c, n_rules2, n_states])

                results[row] = [rule, rule_c, initial_state, pin[0], pin[1], rule2, rule2_c, n_rules2, n_states]
                row += 1

    # PLOT: cell position + value (color) VS % trajectories that fall into largest cycle (y) VS rule (x) VS cycle length
    results = pd.DataFrame.from_dict(results, orient='index', columns=['rule1', 'rule1_c', 'initial_state',
                                                                       'pin_position', 'pin_value', 'rule2',
                                                                       'rule2_c', 'n_rules2', 'n_states'])

    # save
    full_path = _out_results_path + "/" + _out_results_file_name + str(width - 2)
    print("[calculate_results] Saving results to: ", full_path)
    file = open(full_path, 'wb') # Its important to use binary mode
    pickle.dump(results, file)
    file.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    widths = [5]

    #paths for the data that will be generated
    root = "./data"
    patterns_path = root + "/patterns"
    trajectories_path = root + "/trajectories"
    pinned_trajectories_path = root + "/pinned_trajectories"
    pinned_results_path = root + "/pinned_results"
    patterns_1D_path = root + "/patterns/1D"
    patterns_1D_space_path = root + "/patterns/1D/space" # for patterns along the 1D space at a single time step
    patterns_1D_time_path = root + "/patterns/1D/time" # for patterns along the time axis at a single space cell

    for _width in widths:

        print(_width)

        # run pins
        # need to run 6 and 7
        full_path = pinned_trajectories_path + "/pinned_trajectories_" + str(_width)
        if(not os.path.exists(full_path)):
            print("[main] Pinned trajectories not found at '" + full_path + "'.\nGenerating now.")
            pinned_trajectories(_width,
                                _out_pinned_trajectories_path = pinned_trajectories_path)

        # process the results
        full_path = pinned_results_path + "/pinned_results_" + str(_width)
        if(not os.path.exists(full_path)):
            print("[main] Results not found at '" + full_path + "'.\nGenerating now")
            calculate_results(_width,
                            _in_trajectories_path = trajectories_path,
                            _in_pinned_trajectories_path = pinned_trajectories_path,
                            _out_results_path = pinned_trajectories_path)  # rule2 isn't rule2_maxc
        file = open(full_path, 'rb')
        results = pickle.load(file)

        # drop the trivial cases
        results = results.drop(results[results.n_states < 3].index)

        # want to turn this into a heatmap for sure
        heatmap_data = []
        for i in range(1, 5):
            dict = {
                'index': 'Class '+str(i),
                'Class 1': ((results["rule1_c"] == "Class "+str(i)) & (results["rule2_c"] == "Class 1")).sum(),
                'Class 2': ((results["rule1_c"] == "Class "+str(i)) & (results["rule2_c"] == "Class 2")).sum(),
                'Class 3': ((results["rule1_c"] == "Class "+str(i)) & (results["rule2_c"] == "Class 3")).sum(),
                'Class 4': ((results["rule1_c"] == "Class "+str(i)) & (results["rule2_c"] == "Class 4")).sum()
            }
            heatmap_data.append(dict)

        heatmap_data = pd.DataFrame.from_records(heatmap_data, index='index')


        plot = sns.heatmap(heatmap_data, annot=True)
        fig = plot.get_figure()
        full_path = pinned_results_path + "/pinned_results_" + str(_width) + ".png"
        fig.savefig(full_path)
        fig.clf()
