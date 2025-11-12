import networkx as nx
import NetworkFunctions as nf
import scipy as sp
import NetworkClasses as nc
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns

seed = 9102025
num_nodes = 400
low = 0
high = 400

########################################
# Experiment 1 measures of convergence #
########################################
rng = np.random.default_rng(seed=seed)

conv1, conv2, cons1, cons2, deg1, deg2,er1,er2, ci1,ci2 = ([] for i in range(10))

for ii in range(100):
    print(ii)

    net0 = nc.GenerateNet4Xp(num_nodes = num_nodes, low = low, high = high, rng=rng)

    net0.select_and_equate_connected_agents()

    agents = net0.connected_agent_idx

    net1 = copy.deepcopy(net0)
    net2 = copy.deepcopy(net0)



    net1.add_shill_to_agent([agents[0]])
    net2.add_shill_to_agent([agents[1]])

    cons1.append(net1.get_consensus_value())
    cons2.append(net2.get_consensus_value())

    deg1.append(net1.net.degree[agents[0]])
    deg2.append(net2.net.degree[agents[1]])

    #conv1.append(net1.estimate_convergence_time())
    #conv2.append(net2.estimate_convergence_time())

    er1.append(net1.get_expansion_rate())
    er2.append(net2.get_expansion_rate())

    ci1.append(net1.committe_inf(agents[0],1))
    ci2.append(net2.committe_inf(agents[1],1))


path = "/home/hereinlies/Pictures/tempimage.png"



