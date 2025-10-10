"""
General functison for the Complex Networks project
"""

def create_graph(nn, mm = 10 seed =9102025,  graph_type="BA"):
    """
    :param
        nn : number of nodes
        mm : number of edges to attach from a new node to existing nodes (only for BA)
        seed : random seed for reproducibility
        graph_type : type of graph to create ("BA" for Barabási-Albert)
    :return: graph object
    """
    match graph_type:
        case "BA":
            temp = nx.complete_graph(20)
            self.base_network = nx.barabasi_albert_graph(nn, mm, seed=seed, initial_graph=temp)