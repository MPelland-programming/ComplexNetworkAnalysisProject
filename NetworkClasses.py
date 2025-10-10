import networkx as nx
import numpy as np


class NetExperiment:
    def __init__(self, opinion_range = [1,401],num_agents=400, graph_type = "BA",seed=9102025):
        """
        Help

        """
        self.nn = num_agents
        self.seed = seed
        self.vmin = opinion_range[0]
        self.vmax = opinion_range[1]
        self.graph_type = graph_type

        #Generate initial opinions x0 and influence values uu
        self.x0 = np.random.uniform(low=low, high=high, size=nn).astype(int)
        self.uu = np.random.uniform(low=low, high=high, size=nn).astype(int)



    def add_shills(self, num_shills, shill_opinion, shill_influence):
        """
        Add shills to the network by replacing existing nodes with shills.
        """
        if num_shills >= self.nn:
            raise ValueError("Number of shills must be less than the number of agents in the network.")

        # Randomly select nodes to replace with shills
        np.random.seed(self.seed)
        shill_indices = np.random.choice(self.nn, size=num_shills, replace=False)

        # Replace opinions and influence values at selected indices
        self.x0[shill_indices] = shill_opinion
        self.uu[shill_indices] = shill_influence