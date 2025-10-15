import networkx as nx
import numpy as np
import NetworkClasses as nf

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
        self.ori_net = nf.create_graph(nn=self.nn, rng=self.rng, graph_type=self.graph_type)

        # Set up opinions and influence values
        self.ori_x0, self.ori_u = nf.gen_node_features(nn=self.nn, low=self.low, high=self.high, rng=self.rng)


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
        self.connected_agent_idx = self.rng.choice(self.nn, size=num_agents, replace=False)

        if equate:
            a_infl = self.ori_u[self.connected_agent_idx[0]]
            a_opin = self.ori_x0[self.connected_agent_idx[0]]

            self.ori_u[self.connected_agent_idx] = a_infl
            self.ori_x0[self.connected_agent_idx] = a_opin


    def compute_metrics(self):
        # Generate adjacency and A matrices
        self.ori_adj, self.ori_A = nf.gen_network_matrices(self.ori_net, self.ori_u)





class NetExperiment1:
    def __init__(self, opinion_range = [1,401],num_agents=400, graph_type = "BA",seed=9102025):
        """
        Creates an single  instance of experiment 1 where a single shill is added.
        Step 1. Generate a base network
        Step 2. Generate two networks, each with a different shill
        Step 3. Equate the agent's initial opinions and influence values
        Step 4.

        """
        self.nn = num_agents
        self.seed = seed
        self.vmin = opinion_range[0]
        self.vmax = opinion_range[1]
        self.graph_type = graph_type

        #Generate initial opinions x0 and influence values uu
        self.x0 = np.random.uniform(low=low, high=high, size=nn).astype(int)
        self.uu = np.random.uniform(low=low, high=high, size=nn).astype(int)


