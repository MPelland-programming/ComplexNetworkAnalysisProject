"""
General functions for the Complex Networks project
"""
import networkx as nx
import numpy as np
import copy


def create_graph(nn, mm = 10, rng='none',  graph_type="BA"):
    """
    :param
        nn : number of nodes
        mm : number of edges to attach from a new node to existing nodes (only for BA)
        rng : a numpy random generator instance
        graph_type : type of graph to create ("BA" for Barabási-Albert)
    :return: graph object
    """

    match graph_type:
        case "BA":
            temp = nx.complete_graph(20)
            base_network = nx.barabasi_albert_graph(nn, mm, seed=rng, initial_graph=temp)

    return base_network


def gen_node_features(nn, low, high,rng):
    """
    Generate random node features (opinions and influence values)
    :param
        nn : number of nodes
        low : minimum value for uniform distribution
        high : maximum value for uniform distribution
        seed : random seed for reproducibility
    :return: tuple of opinions, influence values
    """
    opinions = rng.integers(low=low, high=high, size=nn)
    influence = rng.integers(low=low, high=high, size=nn)

    return (opinions,influence)


#def compute_lefteigen(A,top = 2):
#    """
#    Compute the normalized left eigenvector(s) of matrix A
#    :param
#        A : matrix A
#        top: number of top eigenvectors to return
#    :return:  normalized left eigenvector
#    """
#    eigval, eigvec = np.linalg.eig(A.T.toarray())
#    idx = np.argsort(-eigval)
#    leigvec = eigvec[:,idx[0:top]].real
#    norm_leigvec = leigvec/np.sum(leigvec,axis=0)

#   return norm_leigvec




def scatter(src,indexes):
    """
    Scatter src values into dest at specified indexes along a given dimension
    Only works for 2D arrays
    :param
        src : source array, column vector
        indexes : indexes to scatter into
    :return: destination array, of size row = src length, col = max indexes with scattered values
    """
    le = src.shape[0]
    dest = np.zeros((le, np.max(indexes)+1))

    for i in range(le):
        dest[i, indexes[i]] = src[i]

    return dest


def add_shills(net, agent_list=[]
               , num_shills=1, num_connection=1
               , x0=[0], influence=[0]
               , shill_opinion=[0], shill_influence=[0]
               , rng='none'
               ):
    """
    Add shills to the network
    :param
        net : networkx graph object, deep copy
        agent_list : list of agents to connect shills to
        num_shills : number of shills to add
        num_connection : number of connections for each shill
        shill_opinion : opinion value for shills ("rand" for random)
        shill_influence : influence value for shills ("rand" for random)
        rng : a numpy random generator instance
    :return: modified networkx graph object
    """

    nn = net.number_of_nodes()
    if num_shills >= nn:
        raise ValueError("Number of shills must be less than the number of agents in the network.")

    #update opinions and influence values
    new_influence = np.append(influence, shill_influence)
    new_x0 = np.append(x0, shill_opinion)

    #Add shill nodes
    net.add_nodes_from(range(nn, nn + num_shills))

    indexes = np.arange(nn)
    rng.shuffle(indexes)

    #if connection_list != []:
    for shill_idx in range(num_shills):
        net.add_edges_from(zip(agent_list[shill_idx], [nn + shill_idx]))


    if num_connection == 1:
        net.add_edges_from(zip(range(nn, nn + num_shills), indexes[0:num_shills]))
    #else:
        #use randomshuffle and the scatter function above

    return (net, new_influence, new_x0)
