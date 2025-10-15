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


def gen_network_matrices(net,influence):
    """
    Generate adjacency and A matrices for the network
    :param
        net : networkx graph object
        influence : array of influence values for each node
    :return: tuple of adjacency matrix, A matrix
    """
    nn = net.number_of_nodes()
    adj = nx.adjacency_matrix(net)

    uurep = np.repeat(np.expand_dims(influence,1),nn,axis=1).T
    denom = influence @ adj
    denom = np.repeat(np.expand_dims(denom,1), nn, axis=1)

    AA = adj*uurep/denom

    return (adj,AA)


def compute_lefteigen(A,top = 2):
    """
    Compute the normalized left eigenvector(s) of matrix A
    :param
        A : matrix A
        top: number of top eigenvectors to return
    :return:  normalized left eigenvector
    """
    eigval, eigvec = np.linalg.eig(A.T.toarray())
    idx = np.argsort(-eigval)
    leigvec = eigvec[:,idx[0:top]].real
    norm_leigvec = leigvec/np.sum(leigvec,axis=0)

    return norm_leigvec


def estimate_convergence_time(A, x0, threshold=0.01, criteria_type = "std",high=400, low=0):
    """
    Estimate the convergence time of the network dynamics
    :param
        A : matrix A
        x0 : initial opinions
        threshold : convergence threshold
        criteria_type : type of convergence criteria ("std" for standard deviation)
    :return: estimated convergence time
    """
    diff = 1+ threshold * (high - low)
    cc = 0
    while diff > 0.01 * (high - low):
        if cc == 0:
            x1 = x0
        else:
            x1 = x2

        x2 = A @ x1
        cc += 1
        match criteria_type:
            case "std":
                diff = np.std(x2 - x1)

    return cc

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


def add_shills(net, num_shills=1, num_connection=1
               , x0=[0], influence=[0]
               , shill_opinion=[0], shill_influence=[0]
               , rng='none'
               ):
    """
    Add shills to the network
    :param
        net : networkx graph object, deep copy
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
    new_influence = np.append[influence, shill_influence]
    new_x0 = np.append[x0, shill_opinion]

    #Add shill nodes
    newnet = copy.deepcopy(net)
    newnet.add_nodes_from(range(nn, nn + num_shills))

    indexes = np.arange(nn)
    rng.shuffle(indexes)

    if num_connection == 1:
        newnet.add_edges_from(zip(range(nn, nn + num_shills), indexes[0:num_shills]))
    #else:
        #use randomshuffle and the scatter function above

    return (newnet, new_influence, new_x0)
