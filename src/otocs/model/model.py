from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import qulacs as qs

from .model_topology import Chain, Lattice


class Model(ABC):
    def __init__(
        self,
        num_qubit: int,
        topology_type: Literal["chain", "lattice"] = "chain",
    ):
        self.num_qubit: int = num_qubit
        self.topology_type: Literal["chain", "lattice"] = topology_type
        self.observable: qs.Observable = qs.Observable(num_qubit)

    @property
    def structure(
        self,
    ) -> Chain | Lattice:
        if self.topology_type == "chain":
            return Chain(self.num_qubit)
        elif self.topology_type == "lattice":
            return Lattice(self.num_qubit)

    @property
    def edges(self):
        return self.structure.edges

    @property
    def nodes(self):
        return self.structure.nodes

    def plot(self, ax=None):
        return self.structure.plot(ax=ax)

    @abstractmethod
    def _add_operator(self, **kwargs) -> None:
        pass

    @abstractmethod
    def add_params(self, **kwargs) -> None:
        pass

    @abstractmethod
    def get_observable(self) -> qs.Observable:
        pass


class ChaoticModel(Model):
    def __init__(
        self,
        num_qubit: int,
        topology_type: Literal["chain", "lattice"] = "chain",
    ):
        self.num_qubit: int = num_qubit
        self.observable: qs.Observable = qs.Observable(num_qubit)
        super().__init__(num_qubit, topology_type)

    def _add_operator(
        self,
        coefficient: float,
        pauli_string: str,
    ) -> None:
        self.observable.add_operator(coefficient, pauli_string)

    def add_params(
        self,
        J_x: float | None = None,
        J_y: float | None = None,
        J_z: float | None = None,
        h_z: float | None = None,
    ) -> None:
        if J_x is None:
            J_x = -0.4
        if J_y is None:
            J_y = -2.0
        if J_z is None:
            J_z = -1.0
        if h_z is None:
            h_z = -0.75

        for i, j in self.edges:
            self._add_operator(J_x, f"X {i} X {j}")
            self._add_operator(J_y, f"Y {i} Y {j}")
            self._add_operator(J_z, f"Z {i} Z {j}")
        for i in self.nodes:
            self._add_operator(h_z, f"Z {i}")

    def get_observable(self) -> qs.Observable:
        return self.observable


class IntegrableModel(Model):
    def __init__(
        self,
        num_qubit: int,
        topology_type: Literal["chain", "lattice"] = "chain",
    ):
        self.num_qubit: int = num_qubit
        self.observable: qs.Observable = qs.Observable(num_qubit)
        super().__init__(num_qubit, topology_type)

    def _add_operator(
        self,
        coefficient: float,
        pauli_string: str,
    ) -> None:
        self.observable.add_operator(coefficient, pauli_string)

    def add_params(
        self,
        J: float,
        D_zz: float,
        h_z: float,
    ) -> None:

        for i, j in self.edges:
            self._add_operator(J, f"X {i} X {j}")
            self._add_operator(J, f"Y {i} Y {j}")
            self._add_operator(J * D_zz, f"Z {i} Z {j}")
        for i in self.nodes:
            self._add_operator(h_z, f"Z {i}")

    def get_observable(self) -> qs.Observable:
        return self.observable


class ZXModel(Model):
    def __init__(
        self,
        num_qubit: int,
        topology_type: Literal["chain", "lattice"] = "chain",
    ):
        self.num_qubit: int = num_qubit
        self.observable: qs.Observable = qs.Observable(num_qubit)
        super().__init__(num_qubit, topology_type)

    def _add_operator(
        self,
        coefficient: float,
        pauli_string: str,
    ) -> None:
        self.observable.add_operator(coefficient, pauli_string)

    def add_params(
        self,
        J_zx: float | None = None,
        h_y: float | None = None,
        topology_type: Literal["chain", "lattice"] = "chain",
    ) -> None:
        if J_zx is None:
            J_zx = 1.0
        if h_y is None:
            h_y = 0.2

        for i, j in self.edges:
            self._add_operator(J_zx, f"Z {i} X {j}")
        for i in self.nodes:
            self._add_operator(h_y, f"Y {i}")

    def get_observable(self) -> qs.Observable:
        return self.observable


class FreeFermionModel(Model):
    def __init__(
        self,
        num_qubit: int,
        topology_type: Literal["chain", "lattice"] = "chain",
    ):
        self.num_qubit: int = num_qubit
        self.observable: qs.Observable = qs.Observable(num_qubit)
        super().__init__(num_qubit, topology_type)

    def _add_operator(
        self,
        coefficient: float,
        pauli_string: str,
    ) -> None:
        self.observable.add_operator(coefficient, pauli_string)

    def add_params(
        self,
        J: float | None = None,
        D_zz: float | None = None,
        h_z: float | None = None,
    ) -> None:
        if J is None:
            J = 1.0
        if D_zz is None:
            D_zz = 0.0
        if h_z is None:
            h_z = 0.0

        for i, j in self.edges:
            self._add_operator(J, f"X {i} X {j}")
            self._add_operator(J, f"Y {i} Y {j}")
            self._add_operator(J * D_zz, f"Z {i} Z {j}")
        for i in self.nodes:
            self._add_operator(h_z, f"Z {i}")

    def get_observable(self) -> qs.Observable:
        return self.observable


class BetheAnsatzModel(Model):
    def __init__(
        self,
        num_qubit: int,
        topology_type: Literal["chain", "lattice"] = "chain",
    ):
        self.num_qubit: int = num_qubit
        self.observable: qs.Observable = qs.Observable(num_qubit)
        super().__init__(num_qubit, topology_type)

    def _add_operator(
        self,
        coefficient: float,
        pauli_string: str,
    ) -> None:
        self.observable.add_operator(coefficient, pauli_string)

    def add_params(
        self,
        J: float | None = None,
        D_zz: float | None = None,
        h_z: float | None = None,
    ) -> None:
        if J is None:
            J = 1.0
        if D_zz is None:
            D_zz = 1.0
        if h_z is None:
            h_z = 0.2

        for i, j in self.edges:
            self._add_operator(J, f"X {i} X {j}")
            self._add_operator(J, f"Y {i} Y {j}")
            self._add_operator(J * D_zz, f"Z {i} Z {j}")
        for i in self.nodes:
            self._add_operator(h_z, f"Z {i}")

    def get_observable(self) -> qs.Observable:
        return self.observable


class MBLModel(Model):
    def __init__(
        self,
        num_qubit: int,
        topology_type: Literal["chain", "lattice"] = "chain",
    ):
        self.num_qubit: int = num_qubit
        self.observable: qs.Observable = qs.Observable(num_qubit)
        super().__init__(num_qubit, topology_type)

    def _add_operator(
        self,
        coefficient: float,
        pauli_string: str,
    ) -> None:
        self.observable.add_operator(coefficient, pauli_string)

    def add_params(
        self,
        coupling_strength: float | None = None,
        bound_field_strength: float | None = None,
        random_seed: int | None = None,
    ) -> None:
        if coupling_strength is None:
            coupling_strength = 1.0
        if bound_field_strength is None:
            bound_field_strength = 5
        if random_seed is None:
            rng = np.random.default_rng(seed=0)
        else:
            rng = np.random.default_rng(seed=random_seed)

        field_strengths = np.zeros(self.num_qubit)

        for i in self.nodes:
            while True:
                x = rng.uniform(-bound_field_strength, bound_field_strength)
                if abs(x) >= coupling_strength:
                    field_strengths[i] = x
                    break
        for i, j in self.edges:
            self._add_operator(coupling_strength, f"X {i} X {j}")
            self._add_operator(coupling_strength, f"Y {i} Y {j}")
            self._add_operator(coupling_strength, f"Z {i} Z {j}")
        for i in self.nodes:
            self._add_operator(field_strengths[i], f"Z {i}")

    def get_observable(self) -> qs.Observable:
        return self.observable


class TransverseIsingModel(Model):
    def __init__(
        self,
        num_qubit: int,
        topology_type: Literal["chain", "lattice"] = "chain",
    ):
        self.num_qubit: int = num_qubit
        self.observable: qs.Observable = qs.Observable(num_qubit)
        super().__init__(num_qubit, topology_type)

    def _add_operator(
        self,
        coefficient: float,
        pauli_string: str,
    ) -> None:
        self.observable.add_operator(coefficient, pauli_string)

    def add_params(
        self,
        coupling_strength: float | None = None,
        field_strength: float | None = None,
    ) -> None:
        if coupling_strength is None:
            coupling_strength = 1.0
        if field_strength is None:
            field_strength = 0.5

        for i, j in self.edges:
            self._add_operator(coupling_strength, f"X {i} X {j}")
        for i in self.nodes:
            self._add_operator(field_strength, f"Z {i}")

    def get_observable(self) -> qs.Observable:
        return self.observable
