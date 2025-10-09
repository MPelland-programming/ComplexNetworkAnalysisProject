import networkx as nx
import numpy as np
import scipy

seed = 9102025
nn = 400
low = 1
high = 400

#Create Barabási-Albert graph
bgraph = nx.complete_graph(20)
n1 = nx.barabasi_albert_graph(nn, 10 , seed=seed, initial_graph=bgraph)

################
#################
n1 = nx.connected_watts_strogatz_graph(nn, 2, 0.3)
############

#Set up parameters
# #opinion
x0 = np.random.uniform(low=low, high=high, size=nn).astype(int)
#influence
uu = np.random.uniform(low=low, high=high, size=nn).astype(int)


#Adjacency matrix
adj = nx.adjacency_matrix(n1)

#A matrix
uurep = np.repeat(np.expand_dims(uu,1),nn,axis=1).T
denom = uu @ adj
denom = np.repeat(np.expand_dims(denom,1), nn, axis=1)

AA = adj*uurep/denom

#eigen stuff
eigval, eigvec = np.linalg.eig(AA.T.toarray())
idx = np.argmax(eigval)
leigvec = eigvec[:,idx].real
norm_leigvec = leigvec/np.sum(leigvec)

#consensus
con = norm_leigvec @ x0

#Estimate convergence time
################## Should I find a real way to do this????
diff = 20
cc = 0
while diff > 0.01*(high-low):
    if cc == 0:
        x1 = x0
    else:
        x1 = x2

    x2 = AA @ x1
    cc += 1

    diff = np.std(x2 - x1)

    #print(cc)
    #print(diff)
print(cc)