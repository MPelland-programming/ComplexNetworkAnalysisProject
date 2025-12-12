import importlib
import networkx as nx
import numpy as np
import scipy as sp
import NetworkFunctions as nf

importlib.reload(nf)

class GenerateNet4Xp:
    """
    Class to generate a network for experiments.
    :param
        num_nodes : number of nodes in the network
        low : minimum value for uniform distribution of opinions and influence values
        high : maximum value for uniform distribution of opinions and influence values
        graph_type : type of graph to create ("BA" for Barabási-Albert)
        rng : a numpy random generator instance
    :return: instance of the class with generated network, opinions, and influence values
    """
    def __init__(self, num_nodes=400
                 , low=0, high=400
                 , graph_type="BA"
                 , rng=np.random.default_rng(seed=1234)
                 ):

        self.nn = num_nodes
        self.low = low
        self.high = high
        self.graph_type = graph_type
        self.rng = rng

        # Create Barabási-Albert graph
        self.net = nf.create_graph(nn=self.nn, rng=self.rng, graph_type=self.graph_type)

        # Set up opinions and influence values
        self.x0, self.u = nf.gen_node_features(nn=self.nn, low=self.low, high=self.high, rng=self.rng)

        self.update_weighted_net()


    def select_and_equate_connected_agents(self, num_agents=2,equate=True):
        """
        Select which agents will be connected to the shills.
        Equate the initial opinions and influence values of the first num_agents agents
        to new_x0 and new_u respectively.
        :param
            num_agents : number of agents to connect to shills
            equate : if True, equate the initial opinions and influence values of the selected agents
        :return: self, index of connected agents and updated opinions and influence values if equate is True
        """
        tmp_idx = self.rng.choice(self.nn, size=num_agents, replace=False)

        if num_agents == 2:
            if self.net.degree[tmp_idx[0]] > self.net.degree[tmp_idx[1]]:
                self.connected_agent_idx = tmp_idx
            else:
                self.connected_agent_idx = tmp_idx[::-1]

        if equate:
            a_infl = self.u[self.connected_agent_idx[0]]
            a_opin = self.x0[self.connected_agent_idx[0]]

            self.u[self.connected_agent_idx] = a_infl
            self.x0[self.connected_agent_idx] = a_opin

        self.update_weighted_net()

    def add_shill_to_agent(self, idx, shill_opinion="agent", shill_influence="agent"):
        """
        Add shills to the original network.
        :param
            shill_opinions : list of opinions for each shill
            shill_influences : list of influence values for each shill
            num_connections : number of connections each shill will have to the original network
        :return: self, updated network, opinions, and influence values with shills added
        """
        if shill_opinion == "agent":
            x0s = self.x0[idx]
        else :
            x0s = shill_opinion
        if shill_influence == "agent":
            us = self.u[idx]
        else:
            us = shill_influence

        self.net, self.u,self.x0 = nf.add_shills(self.net,agent_list=[idx]
                   , num_shills=1, num_connection=1
                   , x0=self.x0, influence=self.u
                   , shill_opinion=x0s, shill_influence=us
                   , rng=self.rng
                   )
        self.nn = self.net.number_of_nodes()

        self.update_weighted_net()

    def update_weighted_net(self):
        """
        Generate the weighted version of the network.
        :return: none, updates self to include a new graph with weighted edges
        """
        #adj = nx.adjacency_matrix(self.net) + sp.sparse.identity(self.nn)

        #uurep = adj*np.repeat(np.expand_dims(self.u, 1), self.nn, axis=1).T
        #denom = self.u @ adj
        #denom = np.repeat(np.expand_dims(denom, 1),self.nn, axis=1)

        #self.wnet = nx.from_scipy_sparse_array(uurep / denom).to_directed()

        adj = nx.adjacency_matrix(self.net).toarray() + np.eye(self.nn)
        uurep = adj * self.u  # Broadcasting
        denom = self.u @ adj
        #denom = (adj * self.u).sum(axis=1)

        self.wnet = nx.from_numpy_array(uurep / denom, create_using=nx.DiGraph)

    def get_consensus_value(self):
        """
        Calculate the consensus value of the network.
        :return: consensus value
        """
        eigvec = np.array(list(
                        nx.eigenvector_centrality(self.wnet, weight="weight").values())
                        )
        norm_eigvec = eigvec/np.sum(eigvec)
        self.consensus = norm_eigvec @ self.x0

        return self.consensus

    def estimate_convergence_time(self,threshold=0.00001, criteria_type="std"):
        """
        Estimate the convergence time of the network dynamics
        :param
            threshold : convergence threshold
            criteria_type : type of convergence criteria ("std" for standard deviation)
        :return: estimated convergence time
        """

        diff = 1 + threshold * (self.high - self.low)
        x1 = self.x0
        cc = 0

        #if not hasattr(self, 'consensus'):
        #    self.get_consensus_value()


        A = nx.adjacency_matrix(self.wnet)

        while diff > threshold * (self.high - self.low):
            x2 = A @ x1
            cc += 1


            match criteria_type:
                case "std":
                    diff = np.std(x2 - x1)

            x1 = x2

        self.estimated_convergence_time = cc
        return cc

    def influence_ratio(self, node):
        """
        Ratio of influence to average influence of neighbors
        :param
            node : target node
        :return: influence ratio
        """
        neighs = list(self.net.neighbors(node))
        avg_infl = np.mean(self.u[neighs]) if neighs else 0

        if avg_infl == 0:
            return 0
        else:
            return self.u[node] / avg_infl




    def get_expansion_rate(self):
        A = nx.adjacency_matrix(self.wnet)
        lams = sp.sparse.linalg.eigsh(A, which="LM", k=2, return_eigenvectors=False)

        return min(lams)

    def get_local_expansion_rate(self, node, n_neigh):
        ulist = self.get_neighbors_at_distance(node, n_neigh)

        subgr = self.wnet.subgraph(ulist)

        A = nx.adjacency_matrix(subgr)
        lams = sp.sparse.linalg.eigsh(A, which="LM", k=2, return_eigenvectors=False)

        return min(lams)

    def get_neighbors_at_distance(self, node, n_neigh):
        """
        Find all neighbors of node at distance n_neigh
        :param n_neigh: distance of neighbors to consider
        :return: list of neighbors at distance n_neigh
        """
        clist = [node.item()]
        for ii in range(n_neigh):
            tmp = []
            for jj in set(clist):
                tmp += list(self.net.neighbors(jj))
            clist += tmp

        ulist = list(set(clist))

        return ulist

    def committe_inf(self,node, n_neigh):
        """
        Implements committe influence by finding all neighbors of node at distance n_neigh
        and using them to compute group degree centrality.
        :param n_neigh: distance of neighbors to consider
        :return: sum of degree-1 of all neighbors of distance n_neigh
        """
        ulist = self.get_neighbors_at_distance(node, n_neigh)


        if len(ulist) < self.nn:
            return nx.group_degree_centrality(self.net, ulist)*(self.nn - len(ulist))
        else:
            return 1

    def committe_dense(self, node, n_neigh):
            """
            Find subgraph density
            :param n_neigh: distance of neighbors to consider
            :return: sum of degree-1 of all neighbors of distance n_neigh
            """
            ulist = self.get_neighbors_at_distance(node, n_neigh)

            return nx.density(self.net.subgraph(ulist))

    def mean_ingoing_to_subgraph(self, node, n_neigh):
        """Calculate mean weight of edges entering subgraph from outside"""
        ulist = self.get_neighbors_at_distance(node, n_neigh)

        subnodes = list(self.wnet.subgraph(ulist).nodes)

        weights = []
        for nod in subnodes:
            for pred in self.wnet.predecessors(nod):
                if pred not in subnodes:  # Edge from outside
                    weights.append(self.wnet[pred][nod]['weight'])
        return sum(weights) / len(weights) if weights else 0

    def mean_outgoing_from_subgraph(self, node, n_neigh):
        """Calculate mean weight of edges leaving subgraph to outside"""
        ulist = self.get_neighbors_at_distance(node, n_neigh)

        subnodes = list(self.wnet.subgraph(ulist).nodes)

        weights = []
        for nod in subnodes:
            for succ in self.wnet.successors(nod):
                if succ not in subnodes:  # Edge to outside
                    weights.append(self.wnet[nod][succ]['weight'])
        return sum(weights) / len(weights) if weights else 0

    def closeness_centrality(self, target_node):
        """
        Calculate average shortest path length from all other nodes to target_node.
        """
        return nx.closeness_centrality(self.net)[target_node]



#class NetExperiment1:
#    def __init__(self, opinion_range = [1,401],num_agents=400, graph_type = "BA",seed=9102025):
#        """
#        Creates an single  instance of experiment 1 where a single shill is added.
#        Step 1. Generate a base network
#        Step 2. Generate two networks, each with a different shill
#        Step 3. Equate the agent's initial opinions and influence values
#        Step 4. add shills to each network
#        Step 5.

 #       """
 #       self.nn = num_agents
 #       self.seed = seed
 #       self.vmin = opinion_range[0]
 #       self.vmax = opinion_range[1]
 #       self.graph_type = graph_type

        #Generate initial opinions x0 and influence values uu
#        self.x0 = np.random.uniform(low=low, high=high, size=nn).astype(int)
#        self.uu = np.random.uniform(low=low, high=high, size=nn).astype(int)


#nn = 3
#u = net1.u[220:223]
#net = net1.net
#adj = (nx.adjacency_matrix(net1.net).toarray() + sp.sparse.identity(net1.nn).toarray())[220:223,220:223]
#adj = nx.adjacency_matrix(net1.net).toarray() + np.eye(net1.nn)
#        uurep = adj * net1.u  # Broadcasting
#        denom = (adj * net1.u).sum(axis=1, keepdims=True)

 #       self.wnet = nx.from_numpy_array(uurep / denom, create_using=nx.DiGraph)