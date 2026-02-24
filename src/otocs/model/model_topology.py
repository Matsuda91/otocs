from abc import ABC, abstractmethod

import networkx as nx
import numpy as np


class Topology(ABC):
    @abstractmethod
    def edges(self):
        pass

    @abstractmethod
    def nodes(self):
        pass


class Lattice(Topology):
    def __init__(self, num_qubit: int):
        self.num_qubit: int = num_qubit

        self._create_graph()

    def _create_graph(self) -> None:
        root = int(np.sqrt(self.num_qubit))
        Ly = None
        for d in range(root, 0, -1):
            if self.num_qubit % d == 0:
                Ly = d
                break
        assert Ly is not None  # N>=1
        Lx = self.num_qubit // Ly
        self._Lx = Lx
        self._Ly = Ly
        self.graph: nx.Graph = nx.grid_2d_graph(Lx, Ly)
        node_list: list[tuple[int, int]] = list(self.graph.nodes())
        self._node_to_qubit = {node: i for i, node in enumerate(node_list)}

    def qubit_index(self, node: tuple[int, int]):
        return self._node_to_qubit[node]

    @property
    def edges(self):

        edges = []
        for u, v in self.graph.edges():
            i = self.qubit_index(u)
            j = self.qubit_index(v)
            edges.append([i, j])
        return edges

    @property
    def nodes(self):
        return list(range(self.num_qubit))


class Chain(Topology):
    def __init__(self, num_qubit: int):
        self.num_qubit: int = num_qubit
        self.graph: nx.Graph = nx.path_graph(num_qubit)
        node_list = list(self.graph.nodes())
        self._node_to_qubit = {node: i for i, node in enumerate(node_list)}

    def edges(self):
        edges = []
        for u, v in self.graph.edges():
            i = self._node_to_qubit[u]
            j = self._node_to_qubit[v]
            edges.append([i, j])
        return edges

    def nodes(self):
        return list(range(self.num_qubit))
