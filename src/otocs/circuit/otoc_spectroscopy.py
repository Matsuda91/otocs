from __future__ import annotations
from typing import Collection
import qulacs as qs
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Literal
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.io as pio
from qulacsvis import circuit_drawer
import numpy as np

from ..style import COLORS


class QuantumStateManager:
    def __init__(self, num_qubit: int):
        self.num_qubit: int = num_qubit
        self.state: qs.QuantumState = qs.QuantumState(num_qubit)

    def set_initial_state(self, state_index: Collection[Literal["0", "1", "+"]]):
        for i, s in enumerate(state_index):
            if s == "0":
                pass
            elif s == "1":
                circuit = qs.QuantumCircuit(self.num_qubit)
                circuit.add_X_gate(i)
                circuit.update_quantum_state(self.state)
            elif s == "+":
                circuit = qs.QuantumCircuit(self.num_qubit)
                circuit.add_H_gate(i)
                circuit.update_quantum_state(self.state)
            else:
                raise ValueError(f"Invalid state index: {s}")

    def get_state(self):
        return self.state


def otoc_circuit(
    observable: qs.Observable,
    targets: tuple[int, int],
    k: int,
    dt: float,
    delta: float | None = None,
    mode: Literal["unitary_check", "otocs"] = "otocs",
) -> qs.QuantumCircuit:
    def calc_repeat(
        observable: qs.Observable,
        dt: float,
        delta: float,
    ) -> int:
        obs_matrix_arr = observable.get_matrix().toarray()
        norm = np.linalg.norm(obs_matrix_arr)
        repeat = int(np.ceil((norm * abs(dt) / delta)))
        print(f"time step       [dt]               : {dt}")
        print(f"observable norm [|H|₂]             : {norm}")
        print(f"Torotter repeat [N: |H|₂*dt/N < δ] : {repeat}")
        return repeat

    if delta is None:
        repeat = 50
    else:
        repeat = calc_repeat(observable, dt, delta)
    num_qubit = observable.get_qubit_count()
    circuit = qs.QuantumCircuit(num_qubit)
    i = targets[0]
    j = targets[1]
    print(i, j)
    for _ in range(int(2 * k)):
        if mode == "otocs":
            circuit.add_Z_gate(j)

        circuit.add_observable_rotation_gate(observable, dt, repeat=repeat)

        if mode == "otocs":
            circuit.add_Z_gate(i)

        circuit.add_observable_rotation_gate(observable, -dt, repeat=repeat)
    return circuit


def execute(
    observable: qs.Observable,
    time_range: np.ndarray,
    targets: tuple[int, int] | None = None,
    echo_k: int | None = None,
    delta: float | None = None,
    initial_state_index: str | Collection[Literal["0", "1", "+"]] = "0",
    mode: Literal["unitary_check", "otocs"] = "otocs",
):
    num_qubit = observable.get_qubit_count()

    if targets is None:
        targets = (0, num_qubit - 1)

    if echo_k is None:
        echo_k = 1

    values = []
    for dt in time_range:
        circuit: qs.QuantumCircuit = otoc_circuit(
            observable=observable,
            targets=targets,
            k=echo_k,
            dt=dt,
            delta=delta,
            mode=mode,
        )

        state_manager = QuantumStateManager(num_qubit)
        if isinstance(initial_state_index, str):
            initial_state_index = [initial_state_index] * num_qubit
        state_manager.set_initial_state(initial_state_index)
        state = state_manager.get_state()
        circuit.update_quantum_state(state)

        values.append(state.get_vector()[0])
    return values


def sweep_echo_k(
    observable: qs.Observable,
    echo_k_range: list[int | float],
    time_range: np.ndarray | None = None,
    targets: tuple[int, int] | None = None,
    delta: float | None = None,
    initial_state_index: str | Collection[Literal["0", "1", "+"]] = "0",
) -> SweepEchoKResult:
    if time_range is None:
        time_range = np.arange(1, 7.01, 0.25)

    results = {}
    for k in tqdm(echo_k_range):
        values = execute(
            observable=observable,
            time_range=time_range,
            targets=targets,
            echo_k=k,
            delta=delta,
            initial_state_index=initial_state_index,
        )
        results[k] = values

    return SweepEchoKResult(
        data={
            "echo_k_range": echo_k_range,
            "time_range": time_range,
            "results": results,
        }
    )


def plot(
    time_range: np.ndarray,
    results: list[complex],
):
    pass


@dataclass
class SweepEchoKResult:
    data: dict[int, list[complex]]

    @property
    def echo_k_range(self) -> list[int]:
        return self.data["echo_k_range"]

    @property
    def time_range(self) -> np.ndarray:
        return self.data["time_range"]

    def get_values(self, echo_k: int) -> list[complex]:
        return self.data["results"][echo_k]

    def plot(self):
        time_range = self.time_range
        echo_k_range = self.echo_k_range

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(f"real", "imaginary"),
            shared_yaxes=True,
        )
        for k in echo_k_range:
            values = self.get_values(k)

            fig.add_trace(
                go.Scatter(
                    x=time_range,
                    y=np.real(values),
                    mode="lines+markers",
                    name=f"k={k}",
                    showlegend=False,
                    marker=dict(color=COLORS[(int(k) - 1) % len(COLORS)]),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=time_range,
                    y=np.imag(values),
                    mode="lines+markers",
                    name=f"k={k}",
                    showlegend=True,
                    marker=dict(color=COLORS[(int(k) - 1) % len(COLORS)]),
                ),
                row=1,
                col=2,
            )

        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(
            title=dict(text="OTOC^(k)"),
            row=1,
            col=1,
        )
        fig.show()
