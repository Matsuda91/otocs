from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Literal

import numpy as np
import plotly.graph_objects as go
import qulacs as qs
from plotly.subplots import make_subplots
from tqdm import tqdm

from otocs.qsp.phiset import QSPPhiSet, TargetFunction

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


def _calc_repeat(
    observable: qs.Observable,
    dt: float,
    delta: float,
    write: bool = True,
) -> int:
    obs_matrix_arr = observable.get_matrix().toarray()
    norm = np.linalg.norm(obs_matrix_arr)
    repeat = int(np.ceil((norm * abs(dt) / delta)))
    if write:
        print("")
        print(f"time step       [dt]               : {dt}")
        print(f"observable norm [|H|₂]             : {norm}")
        print(f"Trotter repeat [N: |H|₂*dt/N < δ]  : {repeat}")
        print("")
    return repeat


def _calc_delta(observable: qs.Observable, dt: float, repeat: int, write: bool = True):
    obs_matrix_arr = observable.get_matrix().toarray()
    norm = np.linalg.norm(obs_matrix_arr)
    delta = np.ceil(repeat(repeat / (norm * abs(dt))))
    if write:
        print("")
        print(f"time step       [dt]             : {dt}")
        print(f"observable norm [|H|₂]           : {norm}")
        print(f"Trotter delta   [δ: N/(|H|₂*dt)] : {repeat}")
        print("")


def qsp_otoc_circuit(
    filter_func: TargetFunction,
    observable: qs.Observable,
    targets: tuple[int, int],
    dt: float,
    delta: float | None = None,
    repeat: float | None = None,
    qsp_polydeg: int | None = None,
    write: bool = True,
):
    if delta is None:
        delta = 0.01

    if repeat is None:
        repeat = _calc_repeat(
            observable,
            dt,
            delta,
            write=write,
        )
    else:
        repeat = repeat
        print("setting repeat overwrites the setted delta as below")
        _calc_delta(
            dt=dt,
            repeat=repeat,
            write=write,
        )

    num_qubit = observable.get_qubit_count()
    circuit = qs.QuantumCircuit(num_qubit)
    i = targets[0]
    j = targets[1]

    qsp_phi_set = QSPPhiSet(
        target_func=filter_func,
        polydeg=qsp_polydeg,
    )
    phi_set_gen = qsp_phi_set.generate(return_phiset=True)
    phi_set = phi_set_gen.get("phiset")
    parity = phi_set_gen.get("parity")
    if parity % 2 == 0:
        raise
    phi_set = np.reshape(phi_set, (len(phi_set) // 2, 2))
    for pdx, phi in enumerate(phi_set):
        circuit.add_RZ_gate(j, phi[0])
        circuit.add_observable_rotation_gate(observable, dt, repeat=repeat)
        circuit.add_RZ_gate(i, phi[1])
        circuit.add_observable_rotation_gate(observable, -dt, repeat=repeat)
    return circuit


def otoc_circuit(
    observable: qs.Observable,
    targets: tuple[int, int],
    k: int,
    dt: float,
    delta: float | None = None,
) -> qs.QuantumCircuit:

    if delta is None:
        repeat = 50
    else:
        repeat = _calc_repeat(observable, dt, delta)
    num_qubit = observable.get_qubit_count()
    circuit = qs.QuantumCircuit(num_qubit)
    i = targets[0]
    j = targets[1]
    for _ in range(int(2 * k)):
        circuit.add_Z_gate(j)
        circuit.add_observable_rotation_gate(observable, dt, repeat=repeat)
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
            "targets": targets,
            "results": results,
        }
    )


@dataclass
class SweepEchoKResult:
    data: dict[int, list[complex]]

    @property
    def echo_k_range(self) -> list[int]:
        return self.data["echo_k_range"]

    @property
    def time_range(self) -> np.ndarray:
        return self.data["time_range"]

    @property
    def targets(self) -> tuple[int, int]:
        return self.data["targets"]

    def get_values(self, echo_k: int) -> list[complex]:
        return self.data["results"][echo_k]

    def plot_phase_distribution(
        self,
        theta_range: np.ndarray | None = None,
        return_figure: bool = False,
    ) -> go.Figure:
        if theta_range is None:
            theta_range = np.linspace(0, np.pi, 100)

        values = []
        for theta in theta_range:
            _values = np.zeros_like(self.time_range)
            for echo_k in self.echo_k_range:
                _values += np.real(self.get_values(echo_k)) * np.cos(2 * echo_k * theta)
            values.append(_values)

        fig = go.Figure(
            data=go.Surface(
                x=self.time_range,
                y=theta_range,
                z=values,
                colorscale="Viridis",
            )
        )
        fig.update_layout(
            title=f"p̃_(i={self.targets[0]},j={self.targets[1]})(θ,t) (surface view)",
            width=700,
            height=500,
            scene=dict(
                xaxis_title="time t",
                yaxis_title="θ",
                zaxis_title="density",
                aspectmode="cube",
            ),
        )
        fig.show()

        if return_figure:
            return fig

    def plot(
        self,
        return_figure: bool = False,
    ):
        time_range = self.time_range
        echo_k_range = self.echo_k_range

        fig1 = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("real", "imaginary"),
            shared_yaxes=True,
        )
        for k in echo_k_range:
            values = self.get_values(k)

            fig1.add_trace(
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
            fig1.add_trace(
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

        fig1.update_xaxes(title_text="Time", row=1, col=1)
        fig1.update_xaxes(title_text="Time", row=1, col=2)
        fig1.update_yaxes(
            title=dict(text="OTOC^(k)"),
            row=1,
            col=1,
        )
        fig1.show()

        fig2 = self.plot_phase_distribution(
            return_figure=True,
        )
        if return_figure:
            return {
                "fig1": fig1,
                "fig2": fig2,
            }


def execute_qsp_otoc(
    observable: qs.Observable,
    filter_func: TargetFunction,
    time_range: np.ndarray,
    targets: tuple[int, int] | None = None,
    initial_state_index: str | Collection[Literal["0", "1", "+"]] = "0",
    delta: float | None = None,
    qsp_polydeg: int | None = None,
    write: bool = False,
):
    num_qubit = observable.get_qubit_count()

    if targets is None:
        targets = (0, num_qubit - 1)

    values = []
    for dt in tqdm(time_range):
        circuit: qs.QuantumCircuit = qsp_otoc_circuit(
            filter_func=filter_func,
            observable=observable,
            targets=targets,
            dt=dt,
            delta=delta,
            qsp_polydeg=qsp_polydeg,
            write=write,
        )

        state_manager = QuantumStateManager(num_qubit)
        if isinstance(initial_state_index, str):
            initial_state_index = [initial_state_index] * num_qubit
        state_manager.set_initial_state(initial_state_index)
        state = state_manager.get_state()
        circuit.update_quantum_state(state)

        values.append(state.get_vector()[0])
    return values
