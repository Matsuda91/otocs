from abc import ABC, abstractmethod
from collections.abc import Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Topology(ABC):
    @abstractmethod
    def edges(self):
        pass

    @abstractmethod
    def nodes(self):
        pass

    @abstractmethod
    def plot(self, ax=None):
        pass

    @abstractmethod
    def remove_nodes(self, targets: list[int]) -> None:
        pass

    @abstractmethod
    def remove_edges(self, targets: list[Sequence[int]]) -> None:
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
        self._qubit_to_node = {i: node for node, i in self._node_to_qubit.items()}

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
        return [self._node_to_qubit[node] for node in self.graph.nodes()]

    def remove_nodes(self, targets: list[int]) -> None:
        for node in targets:
            if not isinstance(node, int):
                raise TypeError("Each node target must be an integer.")
            graph_node = self._qubit_to_node.get(node)
            if graph_node is None or not self.graph.has_node(graph_node):
                raise ValueError(f"Node {node} does not exist in lattice topology.")
            self.graph.remove_node(graph_node)

    def remove_edges(self, targets: list[Sequence[int]]) -> None:
        for edge in targets:
            if len(edge) != 2:
                raise ValueError("Each edge must have exactly two node indices.")
            i, j = int(edge[0]), int(edge[1])
            u = self._qubit_to_node.get(i)
            v = self._qubit_to_node.get(j)
            if u is None or v is None or not self.graph.has_edge(u, v):
                raise ValueError(f"Edge ({i}, {j}) does not exist in lattice topology.")
            self.graph.remove_edge(u, v)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        pos = {node: node for node in self.graph.nodes()}
        labels = {node: self.qubit_index(node) for node in self.graph.nodes()}
        nx.draw(
            self.graph,
            pos=pos,
            ax=ax,
            with_labels=True,
            labels=labels,
            font_weight="bold",
        )
        return ax


class Chain(Topology):
    def __init__(self, num_qubit: int):
        self.num_qubit: int = num_qubit
        self.graph: nx.Graph = nx.path_graph(num_qubit)
        node_list = list(self.graph.nodes())
        self._node_to_qubit = {node: i for i, node in enumerate(node_list)}
        self._qubit_to_node = {i: node for node, i in self._node_to_qubit.items()}

    @property
    def edges(self):
        edges = []
        for u, v in self.graph.edges():
            i = self._node_to_qubit[u]
            j = self._node_to_qubit[v]
            edges.append([i, j])
        return edges

    @property
    def nodes(self):
        return [self._node_to_qubit[node] for node in self.graph.nodes()]

    def remove_nodes(self, targets: list[int]) -> None:
        for node in targets:
            if not isinstance(node, int):
                raise TypeError("Each node target must be an integer.")
            graph_node = self._qubit_to_node.get(node)
            if graph_node is None or not self.graph.has_node(graph_node):
                raise ValueError(f"Node {node} does not exist in chain topology.")
            self.graph.remove_node(graph_node)

    def remove_edges(self, targets: list[Sequence[int]]) -> None:
        for edge in targets:
            if len(edge) != 2:
                raise ValueError("Each edge must have exactly two node indices.")
            i, j = int(edge[0]), int(edge[1])
            u = self._qubit_to_node.get(i)
            v = self._qubit_to_node.get(j)
            if u is None or v is None or not self.graph.has_edge(u, v):
                raise ValueError(f"Edge ({i}, {j}) does not exist in chain topology.")
            self.graph.remove_edge(u, v)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        pos = {node: (node, 0) for node in self.graph.nodes()}
        labels = {node: self._node_to_qubit[node] for node in self.graph.nodes()}
        nx.draw(
            self.graph,
            pos=pos,
            ax=ax,
            with_labels=True,
            labels=labels,
            font_weight="bold",
        )
        return ax
