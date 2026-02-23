from abc import ABC, abstractmethod

import numpy as np


class Topology(ABC):
    @abstractmethod
    def edges(self) -> None:
        pass

    @abstractmethod
    def nodes(self) -> None:
        pass


class Lattice(Topology):
    def __init__(
        self,
        num_qubit: int,
    ):
        self.num_qubit: int = num_qubit
        self._Lx: int = None
        self._Ly: int = None

    @property
    def size(self) -> tuple[int, int]:
        if self._Lx is None or self._Ly is None:
            # pick Ly as the largest divisor <= sqrt(N), then Lx = N // Ly
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
        return (self._Lx, self._Ly)

    @size.setter
    def size(self, x, y):
        if x <= 0 or y <= 0:
            raise ValueError("size must be positive integers (Lx, Ly).")
        if x * y != self.num_qubit:
            raise ValueError(
                f"Lx*Ly ({x * y}) must equal num_qubit ({self.num_qubit})."
            )
        self._Lx = x
        self._Ly = y

    def index(self, x: int, y: int) -> int:
        return self._Lx * y + x  # row-major

    def edges(self) -> None:
        N = self.num_qubit
        Lx, Ly = self.size

        edges = []
        for y in range(Ly):
            for x in range(Lx):
                i = self.index(x, y)

                # right neighbor (x+1, y)
                if x + 1 < Lx:
                    j = self.index(x + 1, y)

                # up/down neighbor (x, y+1)
                if y + 1 < Ly:
                    j = self.index(x, y + 1)
                edges.append([i, j])
        return edges

    def nodes(self) -> None:
        return list(range(self.num_qubit))


class Chain(Topology):
    def __init__(self, num_qubit: int):
        self.num_qubit: int = num_qubit

    def edges(self) -> None:
        edges = []
        for i in range(self.num_qubit - 1):
            edges.append([i, i + 1])
        return edges

    def nodes(self) -> None:
        return list(range(self.num_qubit))
