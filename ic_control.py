import pickle
import numpy as np
from itertools import product
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

    for rule in rules:

        trajectories[rule] = {}

        for initial_state in initial_states:

            trajectories[rule][str(initial_state)] = CA_run(initial_state, 2**width, rule)

    # save
    file = open('trajectories', 'wb')
    pickle.dump(trajectories, file)
    file.close()


def eca_state_space(width):

    G = nx.MultiDiGraph()
    rules = [30, 45, 73, 110, 150, 105, 54, 22, 60, 146, 126, 62, 90, 18, 122, 26, 154, 94, 41, 57, 156, 28, 58, 78,
             178, 77, 50, 13, 25, 37, 9, 35, 106, 3, 27, 43, 184, 56, 11, 142, 14, 134, 24, 7, 152, 170, 46, 15, 33,
             1, 42, 162, 6, 5, 138, 38, 10, 74, 34, 29, 130, 2, 204, 200, 172, 108, 76, 72, 51, 44, 104, 232, 140, 132,
             23, 12, 164, 36, 19, 4, 168, 40, 160, 136, 128, 32, 8, 0]
    initial_states = [np.array(bits) for bits in product([0, 1], repeat=width)]

    # loops over all rules
    for rule in rules:

        # loop over all initial states
        for initial_state in initial_states:

            data = CA_run(initial_state, 2, rule)

            # add edge to the multigraph
            G.add_edge(str(data[0]), str(data[1]), rule=rule)

    # add entropy values to all the states
    #for initial_state in initial_states:
    #    G.nodes[np.array2string(initial_state)]["entropy"] = entropy(initial_state, base=2)

    # save
    file = open('state_space', 'wb')
    pickle.dump(G, file)
    file.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    widths = [5]

    for width in widths:

        # regular ECA state space over all the rules
        #eca_state_space(width)

        # all ECA trajectories up to 2^w
        #eca_trajectories(width)

        rules = [30, 45, 73, 110, 150, 105, 54, 22, 60, 146, 126, 62, 90, 18, 122, 26, 154, 94, 41, 57, 156, 28, 58, 78,
             178, 77, 50, 13, 25, 37, 9, 35, 106, 3, 27, 43, 184, 56, 11, 142, 14, 134, 24, 7, 152, 170, 46, 15, 33,
             1, 42, 162, 6, 5, 138, 38, 10, 74, 34, 29, 130, 2, 204, 200, 172, 108, 76, 72, 51, 44, 104, 232, 140, 132,
             23, 12, 164, 36, 19, 4, 168, 40, 160, 136, 128, 32, 8, 0]
        initial_states = [np.array(bits) for bits in product([0, 1], repeat=width)]

        file = open("state_space", 'rb')
        G = pickle.load(file)

        file = open("trajectories", 'rb')
        trajectories = pickle.load(file)

        # look at a single cell in the IC
        # when that cell is 0 or 1, does the trajectory fall into the largest cycle for that rule?

        results = {}
        row = 0

        # loop over rules
        for rule in rules:

            # loop over position in IC
            for ic_position in range(0, width):

                # loop over values (just [0, 1])
                for value in [0, 1]:

                    # the subgraph of that rule only
                    G_rule = nx.DiGraph([(u, v, d) for u, v, d in G.edges(data=True) if d['rule'] == rule])

                    # for that rule, find the largest cycle
                    cycles = sorted(list(nx.simple_cycles(G_rule)), key = lambda s: len(s))
                    largest_cycle = cycles[-1]
                    largest_cycle_size = len(largest_cycle)

                    # find ICs with pos + value
                    controller_ics = list(filter(lambda x: x[ic_position] == value, initial_states))

                    # when the cell is always position + VALUE, what % of trajectories fall into largest cycle?
                    frac_ic = sum([1 if str(trajectories[rule][str(controller_ic)][-1]) in largest_cycle else 0 for controller_ic in controller_ics]) / len(controller_ics)

                    # save the results
                    results[row] = [rule, largest_cycle_size, ic_position, value, frac_ic]
                    row += 1

        # PLOT: cell position + value (color) VS % trajectories that fall into largest cycle (y) VS rule (x) VS cycle length
        results = pd.DataFrame.from_dict(results, orient='index', columns=['rule', 'largest_cycle_size', 'ic_position', 'value', 'frac_ic'])

        # want to get a rank order of the rules
        #results['ranked_rule'] = results['frac_ic'].rank()
        results = results.sort_values('frac_ic', ignore_index=True).reset_index()

        sns.set_style("whitegrid")
        plot = sns.relplot(data=results, x='index', y="frac_ic", size="largest_cycle_size", hue="largest_cycle_size",
                           sizes=(40, 400), edgecolor='none', alpha=0.5, aspect=1).set(title=str(width))
        sns.despine(left=True, bottom=True)
        plot.savefig("plot_" + str(width) + ".png")

    # now with all possible single-cell pins
    #pinned_state_space(width)
