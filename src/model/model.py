import qulacs as qs
import numpy as np
from typing import Literal
from abc import ABC, abstractmethod


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


class Observable:
    def __init__(self, num_qubit: int):
        self.num_qubit: int = num_qubit
        self.observable: qs.Observable = qs.Observable(num_qubit)

    def add_operator(
        self,
        coefficient: float,
        pauli_string: str,
    ) -> None:
        self.observable.add_operator(coefficient, pauli_string)

    def add_chaotic_model(
        self,
        J_x: float | None = None,
        J_y: float | None = None,
        J_z: float | None = None,
        h_z: float | None = None,
        topology_type: Literal["chain", "lattice"] = "chain",
    ) -> None:
        if J_x is None:
            J_x = -0.4
        if J_y is None:
            J_y = -2.0
        if J_z is None:
            J_z = -1.0
        if h_z is None:
            h_z = -0.75
        if topology_type == "chain":
            structure = Chain(self.num_qubit)
        elif topology_type == "lattice":
            structure = Lattice(self.num_qubit)
        for i, j in structure.edges():
            self.add_operator(J_x, f"X {i} X {j}")
            self.add_operator(J_y, f"Y {i} Y {j}")
            self.add_operator(J_z, f"Z {i} Z {j}")
        for i in structure.nodes():
            self.add_operator(h_z, f"Z {i}")

    def add_integrable_model(
        self,
        J: float,
        D_zz: float,
        h_z: float,
        topology_type: Literal["chain", "lattice"] = "chain",
    ) -> None:
        if topology_type == "chain":
            structure = Chain(self.num_qubit)
        elif topology_type == "lattice":
            structure = Lattice(self.num_qubit)
        for i, j in structure.edges():
            self.add_operator(J, f"X {i} X {j}")
            self.add_operator(J, f"Y {i} Y {j}")
            self.add_operator(J * D_zz, f"Z {i} Z {j}")
        for i in structure.nodes():
            self.add_operator(h_z, f"Z {i}")

    def add_zx_model(
        self,
        bound_strength: float | None = None,
        topology_type: Literal["chain", "lattice"] = "chain",
    ):
        N = self.num_qubit
        if bound_strength is None:
            bound_strength = 1.0
        rng = np.random.default_rng(seed=0)
        if topology_type == "chain":
            structure = Chain(N)
        elif topology_type == "lattice":
            structure = Lattice(N)

        for i, j in structure.edges():
            strength = rng.uniform(-bound_strength, bound_strength)
            self.add_operator(strength, f"Z {i} X {j}")

    def add_free_fermion_model(
        self,
        J: float | None = None,
        D_zz: float | None = None,
        h_z: float | None = None,
    ):
        if J is None:
            J = 1.0
        if D_zz is None:
            D_zz = 0.0
        if h_z is None:
            h_z = 0.0
        self.add_integrable_model(
            J=J,
            D_zz=D_zz,
            h_z=h_z,
        )

    def add_bethe_ansatz_model(
        self,
        J: float | None = None,
        D_zz: float | None = None,
        h_z: float | None = None,
    ):
        if J is None:
            J = 1.0
        if D_zz is None:
            D_zz = 1.0
        if h_z is None:
            h_z = 0.2
        self.add_integrable_model(
            J=J,
            D_zz=D_zz,
            h_z=h_z,
        )

    def add_mbl_model(
        self,
        coupling_strength: float | None = None,
        bound_field_strength: float | None = None,
        random_seed: int | None = None,
    ):
        if coupling_strength is None:
            coupling_strength = 1.0
        if bound_field_strength is None:
            bound_field_strength = 5
        if random_seed is None:
            rng = np.random.default_rng(seed=0)
        else:
            rng = np.random.default_rng(seed=random_seed)

        field_strengths = np.zeros(self.num_qubit)

        structure = Chain(self.num_qubit)
        for i in structure.nodes():
            while True:
                x = rng.uniform(-bound_field_strength, bound_field_strength)
                if abs(x) >= coupling_strength:
                    field_strengths[i] = x
                    break
        for i, j in structure.edges():
            self.add_operator(coupling_strength, f"X {i} X {j}")
            self.add_operator(coupling_strength, f"Y {i} Y {j}")
            self.add_operator(coupling_strength, f"Z {i} Z {j}")
        for i in structure.nodes():
            self.add_operator(field_strengths[i], f"Z {i}")

    def add_transverse_ising_model(
        self,
        coupling_strength: float = 1.0,
        field_strength: float = 0.5,
        topology_type: Literal["chain", "lattice"] = "chain",
    ) -> None:
        if topology_type == "chain":
            structure = Chain(self.num_qubit)
        elif topology_type == "lattice":
            structure = Lattice(self.num_qubit)

        for i, j in structure.edges():
            self.add_operator(coupling_strength, f"X {i} X {j}")
        for i in structure.nodes():
            self.add_operator(field_strength, f"Z {i}")

    def get_observable(self) -> qs.Observable:
        return self.observable
