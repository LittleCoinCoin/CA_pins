import pickle
import numpy as np
from itertools import product
import copy
import networkx as nx
import seaborn as sns
import pandas as pd

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

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


def eca_trajectories(width):

    trajectories = {}

    rules = [30, 45, 73, 110, 150, 105, 54, 22, 60, 146, 126, 62, 90, 18, 122, 26, 154, 94, 41, 57, 156, 28, 58, 78,
             178, 77, 50, 13, 25, 37, 9, 35, 106, 3, 27, 43, 184, 56, 11, 142, 14, 134, 24, 7, 152, 170, 46, 15, 33,
             1, 42, 162, 6, 5, 138, 38, 10, 74, 34, 29, 130, 2, 204, 200, 172, 108, 76, 72, 51, 44, 104, 232, 140, 132,
             23, 12, 164, 36, 19, 4, 168, 40, 160, 136, 128, 32, 8, 0]

    initial_states = [np.array(bits) for bits in product([0, 1], repeat=width)]

    nb_steps = 2**width
    for rule in rules:

        trajectories[rule] = {}

        for initial_state in initial_states:

            trajectories[rule][str(initial_state)] = CA_run(initial_state, nb_steps, rule)

    # save
    file = open('trajectories_' + str(width), 'wb')
    pickle.dump(trajectories, file)
    file.close()


def pinned_trajectories(width):

    trajectories = {}

    rules = [30, 45, 73, 110, 150, 105, 54, 22, 60, 146, 126, 62, 90, 18, 122, 26, 154, 94, 41, 57, 156, 28, 58, 78,
             178, 77, 50, 13, 25, 37, 9, 35, 106, 3, 27, 43, 184, 56, 11, 142, 14, 134, 24, 7, 152, 170, 46, 15, 33,
             1, 42, 162, 6, 5, 138, 38, 10, 74, 34, 29, 130, 2, 204, 200, 172, 108, 76, 72, 51, 44, 104, 232, 140, 132,
             23, 12, 164, 36, 19, 4, 168, 40, 160, 136, 128, 32, 8, 0]

    initial_states = [np.array(bits) for bits in product([0, 1], repeat=width)]

    # 2 pin positions
    pins = list(product(product(range(0, width), range(0, width)), product(range(0, 2), range(0, 2))))  # (position, value)

    # loops over all rules
    for rule in rules:

        trajectories[rule] = {}

        # loop over all single pins
        for pin in pins:

            trajectories[rule][str(pin)] = {}

            # loop over all initial states
            for initial_state in initial_states:

                trajectories[rule][str(pin)][str(initial_state)] = {}

                # put pins in
                initial_state_pinned = copy.deepcopy(initial_state)
                initial_state_pinned[pin[0][0]] = pin[1][0]
                initial_state_pinned[pin[0][1]] = pin[1][1]
                trajectories[rule][str(pin)][str(initial_state)][0] = initial_state_pinned

                t0 = initial_state_pinned

                # put the pin in for every time step
                for t in range(1, 2 ** width):

                    data = CA_run(t0, 2, rule)
                    t1 = data[-1]

                    # put pin in
                    t1[pin[0][0]] = pin[1][0]
                    t1[pin[0][1]] = pin[1][1]

                    trajectories[rule][str(pin)][str(initial_state)][t] = t1

                    t0 = t1

    # Its important to use binary mode
    file = open('pinned_trajectories_' + str(width), 'wb')

    # source, destination
    pickle.dump(trajectories, file)
    file.close()


def calculate_results(width):

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

    rules = [30, 45, 73, 110, 150, 105, 54, 22, 60, 146, 126, 62, 90, 18, 122, 26, 154, 94, 41, 57, 156, 28, 58, 78,
             178, 77, 50, 13, 25, 37, 9, 35, 106, 3, 27, 43, 184, 56, 11, 142, 14, 134, 24, 7, 152, 170, 46, 15, 33,
             1, 42, 162, 6, 5, 138, 38, 10, 74, 34, 29, 130, 2, 204, 200, 172, 108, 76, 72, 51, 44, 104, 232, 140, 132,
             23, 12, 164, 36, 19, 4, 168, 40, 160, 136, 128, 32, 8, 0]

    # regular ECA state space over all the rules
    # pinned_trajectories(width)

    # all ECA trajectories up to 2^w
    eca_trajectories(width - 2)

    file = open('pinned_trajectories_' + str(width), 'rb')
    pinned_trajectories = pickle.load(file)

    file = open('trajectories_' + str(width - 2), 'rb')
    trajectories = pickle.load(file)

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

    # Its important to use binary mode
    file = open('pinned_results_' + str(width), 'wb')

    # source, destination
    pickle.dump(results, file)
    file.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    widths = [5]

    for width in widths:

        print(width)

        # run pins
        #pinned_trajectories(width)  # need to run 6 and 7

        # process the results
        #calculate_results(width)  # rule2 isn't rule2_maxc

        file = open('pinned_results_' + str(width), 'rb')
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
        fig.savefig("pinned_results_" + str(width) + ".png")
        fig.clf()
