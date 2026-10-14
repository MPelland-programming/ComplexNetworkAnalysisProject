# Complex Network Analysis Project

A Python simulation study of **opinion dynamics on social networks**, and how adding a single high-influence "shill" node to an agent changes how quickly the network reaches consensus. Built for a course on complex networks.

## Overview

The project generates scale-free (Barabási–Albert) networks of agents. Each agent has an initial opinion and an influence value. Opinions evolve by repeatedly averaging over an influence-weighted version of the network until they converge.

The main experiment compares two otherwise-identical agents:

1. Generate a network of 400 agents with random opinions and influence values.
2. Pick two agents and give them the same initial opinion and influence (the higher-degree agent is listed first).
3. Make three copies of the network: a baseline, one with a shill attached to agent 1, and one with a shill attached to agent 2. Each shill copies its agent's opinion and has an influence of 200.
4. Estimate convergence time for all three, then relate the differences to structural properties of the network around each agent.

This is repeated over 100 random networks (seed `9102025`).

## Repository structure

| File | Description |
| --- | --- |
| `main.py` | Runs Experiment 1: generates (or loads) the simulations and computes the per-simulation metrics. |
| `NetworkClasses.py` | `GenerateNet4Xp`, the class that builds a network and implements the dynamics and metrics. |
| `NetworkFunctions.py` | Helpers: graph generation, random node features, adding shills, plotting. |
| `simulations.npy` | Saved results of 100 simulations (each a tuple of baseline, shill-on-agent-1, shill-on-agent-2 networks). |
| `simulations_inf_3000.npy` | Saved simulation set, apparently a variant with shill influence of 3000 (judging by the file name). |
| `test.py` | Scratch file. Not part of the analysis. |
| `LICENSE` | MIT license. |

## The model

- **Graph:** Barabási–Albert graph with `num_nodes = 400`, attaching 10 edges per new node, seeded with a complete graph of 20 nodes.
- **Node features:** each agent gets an integer opinion `x0` and influence `u`, drawn uniformly from `[low, high)` (default 0 to 400).
- **Weighted network:** the adjacency matrix (with self-loops) is weighted by influence and normalised to produce a directed weighted graph, `wnet`.
- **Dynamics:** opinions are updated by repeated multiplication with the weighted adjacency matrix, `x(t+1) = A x(t)`.
- **Convergence time:** the number of iterations until the standard deviation of `x(t+1) - x(t)` falls below `threshold * (high - low)` (default threshold `1e-5`; `main.py` uses `1e-4`).
- **Shills:** an extra node attached to a single agent, with a chosen opinion and influence.

## Metrics computed

`GenerateNet4Xp` provides the following, mostly evaluated around the targeted agent:

- **Convergence time** (`estimate_convergence_time`) and the consensus value (`get_consensus_value`, via eigenvector centrality of the weighted network).
- **Expansion rate** (`get_expansion_rate`, `get_local_expansion_rate`), the second eigenvalue of the weighted adjacency matrix, globally and on a local neighbourhood.
- **Committee influence and density** (`committe_inf`, `committe_dense`): group degree centrality and subgraph density of an agent's neighbourhood.
- **Boundary weights** (`mean_ingoing_to_subgraph`, `mean_outgoing_from_subgraph`): mean weight of edges entering and leaving the neighbourhood.
- **Influence ratio** (`influence_ratio`): an agent's influence relative to the mean of its neighbours.
- **Closeness centrality** (`closeness_centrality`).

`main.py` also computes effective graph resistance and eigenvector centrality using NetworkX.

## Requirements

- Python 3.10 or newer (the code uses `match` statements)
- `numpy`
- `scipy`
- `networkx` (a recent version, since `effective_graph_resistance` is used)
- `matplotlib`
- `seaborn`

```bash
pip install numpy scipy networkx matplotlib seaborn
```

## Usage

Clone the repository and run the experiment from its root directory:

```bash
git clone https://github.com/MPelland-programming/ComplexNetworkAnalysisProject.git
cd ComplexNetworkAnalysisProject
python main.py
```

Two flags near the top of the experiment section in `main.py` control behaviour:

- `gen_simul = True` generates 100 new simulations. Set it to `False` to load `simulations.npy` instead and skip the (slow) generation step.
- `saving = False` set to `True` writes newly generated simulations to `simulations.npy`.

To load the saved data in your own code, `NetworkClasses.py` must be importable, because the files contain pickled `GenerateNet4Xp` objects:

```python
import numpy as np
import NetworkClasses  # needed so the pickled objects can be reconstructed

sims = np.load("simulations.npy", allow_pickle=True)
baseline, shill_a1, shill_a2 = sims[0]
print(shill_a1.estimated_convergence_time)
```

Only load `.npy` files with `allow_pickle=True` from sources you trust.

### Notes

- `NetworkFunctions.py` and `main.py` contain output paths hard-coded to `/home/hereinlies/Pictures/`. Change these before using the plotting helpers (`scatterplot`, `nice_scatter`).
- Results are reproducible through the fixed seed in `main.py`.

## License

Released under the [MIT License](LICENSE).
