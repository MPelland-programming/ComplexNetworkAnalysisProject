import importlib

import networkx as nx
import NetworkFunctions as nf
import scipy as sp
import NetworkClasses as nc
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

importlib.reload(nc)

seed = 9102025
num_nodes = 400
low = 0
high = 400

########################################
# Experiment 1 measures of convergence #
########################################
rng = np.random.default_rng(seed=seed)

list_simul = []

gen_simul = True
saving = False
if gen_simul:
    for ii in range(100):
        print(ii)

        net0 = nc.GenerateNet4Xp(num_nodes = num_nodes, low = low, high = high, rng=rng)

        net0.select_and_equate_connected_agents()

        agents = net0.connected_agent_idx

        net1 = copy.deepcopy(net0)
        net2 = copy.deepcopy(net0)

        net1.add_shill_to_agent([agents[0]], shill_influence=200)
        net2.add_shill_to_agent([agents[1]], shill_influence=200)

        net0.estimate_convergence_time(threshold=0.0001)
        net1.estimate_convergence_time(threshold=0.0001)
        net2.estimate_convergence_time(threshold=0.0001)

        list_simul.append((net0, net1, net2))

    # sav list_simul to file
    if saving:
        #with open('file.pkl', 'wb') as f:
        #    pickle.dump(list_simul, f)
        np.save("simulations.npy", list_simul, allow_pickle=True)
else:
    # load list_simul from file
    list_simul = np.load("simulations.npy", allow_pickle=True)


conv1, conv2, cons1, cons2, deg1, deg2,er1,er2, ci11,ci21,ci10,ci20,dens10,dens20,dens11,dens21, meanin1,meanout1, meanin2,meanout2, localer1, localer2, hc1, hc2= (np.empty(0) for i in range(24))

kir1, kir2, wkir1, wkir2= (np.empty(0) for i in range(4))

cc = 0
for ss in list_simul:
    print(cc)
    cc += 1
    agents = ss[0].connected_agent_idx

    #cons1.append(ss[1].get_consensus_value())
    #cons2.append(ss[2].get_consensus_value())

    deg1= np.append(deg1, ss[1].net.degree[agents[0]])
    deg2=np.append(deg2, ss[1].net.degree[agents[1]])

    conv1 = np.append(conv1, ss[1].estimated_convergence_time)
    conv2 = np.append(conv2, ss[2].estimated_convergence_time)

    er1 = np.append(er1, ss[1].get_expansion_rate())
    er2 = np.append(er2,ss[2].get_expansion_rate())

    ci11 = np.append(ci11,ss[1].committe_inf(agents[0],1))
    ci21 = np.append(ci21,ss[2].committe_inf(agents[1],1))

    ci10 = np.append(ci10,ss[1].committe_inf(agents[0],0))
    ci20 = np.append(ci20,ss[2].committe_inf(agents[1],0))

    dens11 = np.append(dens11,ss[1].committe_dense(agents[0],1))
    dens21 = np.append(dens21,ss[2].committe_dense(agents[1],1))

    dens10 = np.append(dens10,ss[1].committe_dense(agents[0],0))
    dens20 = np.append(dens20,ss[2].committe_dense(agents[1],0))

    meanin1 = np.append(meanin1,ss[0].mean_ingoing_to_subgraph(agents[0],1))
    meanin2 = np.append(meanin2, ss[0].mean_ingoing_to_subgraph(agents[1], 1))

    meanout1 = np.append(meanout1,ss[0].mean_outgoing_from_subgraph(agents[0],1))
    meanout2 = np.append(meanout2, ss[0].mean_outgoing_from_subgraph(agents[1], 1))

    localer1 = np.append(localer1,ss[1].get_local_expansion_rate(agents[0],1))
    localer2 = np.append(localer2,ss[2].get_local_expansion_rate(agents[1],1))

    hc1 = np.append(hc1, ss[1].closeness_centrality(agents[0]))
    hc2 = np.append(hc2, ss[2].closeness_centrality(agents[1]))

    kir1 = np.append(kir1, nx.effective_graph_resistance(ss[1].net))
    kir2 = np.append(kir2, nx.effective_graph_resistance(ss[2].net))

meanin10,meanin20,meanout10,meanout20,infl1,infl2 = (np.empty(0) for i in range(6))
cc = 0
for ss in list_simul:
    print(cc)
    cc += 1
    agents = ss[0].connected_agent_idx

    meanin10 = np.append(meanin10,ss[0].mean_ingoing_to_subgraph(agents[0],0))
    meanin20 = np.append(meanin20, ss[0].mean_ingoing_to_subgraph(agents[1], 0))

    meanout10 = np.append(meanout10,ss[0].mean_outgoing_from_subgraph(agents[0],0))
    meanout20 = np.append(meanout20, ss[0].mean_outgoing_from_subgraph(agents[1], 0))

    infl1 = np.append(infl1,ss[0].influence_ratio(agents[0]))
    infl2 = np.append(infl2, ss[0].influence_ratio(agents[1]))

path = "/home/hereinlies/Pictures/tempimage.png"




influence= []
for ss in list_simul:
        agents = ss[0].connected_agent_idx
        influence.append(ss[0].u[agents[0]])

kir0 = np.empty(0)
cc = 0
for ss in list_simul:
    print(cc)
    cc += 1
    kir0 = np.append(kir0, nx.effective_graph_resistance(ss[0].net))



weig1 = np.empty(0)
weig2 = np.empty(0)
cc = 0
for ss in list_simul:
    print(cc)
    cc += 1
    weig1 = np.append(weig1, nx.eigenvector_centrality(ss[1].wnet, weight = "weight")[agents[0]])
    weig2 = np.append(weig2, nx.eigenvector_centrality(ss[2].wnet, weight = "weight")[agents[1]])