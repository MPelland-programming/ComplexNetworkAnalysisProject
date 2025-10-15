import NetworkFunctions as nf
import numpy as np

seed = 9102025
num_nodes = 400
low = 0
high = 400

########################################
# Experiment 1 measures of convergence #
########################################
rng = np.random.default_rng(seed=seed)



#Create Barabási-Albert graph
ori_net = nf.create_graph(nn=nn, rng=rng, graph_type="BA") #### check how to pass seed

#Set up opinions and influence values
ori_x0, ori_u = nf.gen_node_features(nn=nn, low=low, high=high, rng=rng)

#add shills
full_net,full_u,full_x0  = nf.add_shills(ori_net
               , num_shills=1, num_connection=1
               , x0=ori_x0, influence=ori_u
               , shill_opinion=[200], shill_influence=[200]
               , rng=rng
               )

#Generate adjacency and A matrices
full_adj,full_A = nf.gen_network_matrices(full_net, full_u)

#Get left eigenvector
leig = nf.compute_lefteigen(full_A,top = 1)

#consensus
con = leig @ full_x0

#Estimate convergence time
################## Should I find a real way to do this????
#contime = estimate_convergence_time(A, x0, threshold=0.01)