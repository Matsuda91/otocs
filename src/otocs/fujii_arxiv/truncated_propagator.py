import numpy as np
import scipy.linalg as linalg
from numpy.polynomial.chebyshev import chebval
from typing import Literal
import qulacs as qs
import plotly.colors as pc
import plotly.graph_objects as go
from collections import defaultdict
from tqdm import tqdm


def indices_bit_is_zero(
    N: int,
    q: int,
    bit_type: Literal["lsb", "msb"] = "lsb",
) -> np.ndarray:
    if bit_type == "lsb":
        q_bit = q
    elif bit_type == "msb":
        q_bit = N - 1 - q
    else:
        raise ValueError("bit_type must be 'lsb' or 'msb'")
    dim = 2**N
    idx = np.arange(dim, dtype=np.int64)
    return idx[((idx >> q_bit) & 1) == 0]


def truncated_propagator_A(U: np.ndarray, N: int, i: int, j: int) -> np.ndarray:
    rows = indices_bit_is_zero(N, i)  # <0_i|
    cols = indices_bit_is_zero(N, j)  # |0_j>
    return U[np.ix_(rows, cols)]  # (2^(N-1), 2^(N-1))


def prepare_time_evolver(H: np.ndarray):
    E, V = linalg.eigh(H)
    Vdag = V.conj().T

    def U(t: float) -> np.ndarray:
        phase = np.exp(-1j * E * t)
        return (V * phase) @ Vdag

    return U


def singular_values_and_thetas(A: np.ndarray):
    result = np.linalg.svd(
        a=A,
    )
    lam = np.clip(result.S, 0.0, 1.0)
    theta = 2.0 * np.arccos(lam)
    return lam, theta


def chebyshev_Tn(lam: np.ndarray, n: int) -> np.ndarray:
    c = np.zeros(n + 1)
    c[n] = 1.0
    return chebval(lam, c)


def moment_M_4k(lam: np.ndarray, k: int) -> float:
    order = int(4 * k)
    return float(np.mean(chebyshev_Tn(lam, order)))


def is_hermitian(observable: qs.Observable) -> bool:
    if not observable.is_hermitian():
        raise ValueError("Observable must be Hermitian.")


def compute_fig2_moments(
    observable: qs.Observable,
    times: np.ndarray,
    i: int = 0,
    js: list[int] | None = None,
    k: int | None = None,  # 4k = 2,4,8,12  (k=1/2,1,2,3)
):
    is_hermitian(observable)
    H = observable.get_matrix().toarray()
    N = int(np.log2(H.shape[0]))
    assert H.shape == (2**N, 2**N)
    assert len(js) == N

    if k is None:
        k = 2

    if js is None:
        js = list(range(N))  # j=0..N-1

    U_of_t = prepare_time_evolver(H)
    results = defaultdict(list)
    for _, j in tqdm(list(enumerate(js))):
        res = []
        for _, t in enumerate(times):
            U = U_of_t(float(t))
            A = truncated_propagator_A(U, N, i=i, j=j)
            lam, _ = singular_values_and_thetas(A)
            res.append(moment_M_4k(lam, k=k))
        results[j] = res

    j_min = min(js)
    j_max = max(js)

    fig = go.Figure()
    for _, j in enumerate(js):
        j_norm = (j - j_min) / (j_max - j_min)
        color = pc.sample_colorscale("Viridis", j_norm)[0]

        fig.add_trace(
            go.Scatter(
                x=times,
                y=results[j],
                mode="lines",
                name=f"j={j}",
                line=dict(color=color),
            )
        )
    fig.update_layout(
        title=f"⟨T_{int(4 * k)}(λ)⟩ for A_(i={i},j)(t)",
        xaxis_title="time t",
        yaxis_title=f"⟨T_{int(4 * k)}(λ)⟩",
    )
    fig.show()


def compute_fig1_theta_histograms(
    observable: qs.Observable,
    times: np.ndarray,
    i: int,
    j: int,
    nbins: int = 80,
):
    is_hermitian(observable)

    num_qubit = observable.get_qubit_count()
    if not (0 <= i < num_qubit):
        raise ValueError(f"i must be in [0, {num_qubit - 1}]")
    if not (0 <= j < num_qubit):
        raise ValueError(f"j must be in [0, {num_qubit - 1}]")

    H = observable.get_matrix().toarray()

    N = int(np.log2(H.shape[0]))
    U_of_t = prepare_time_evolver(H)

    theta_edges = np.linspace(0.0, np.pi, nbins + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    # densities = np.zeros((len(times), nbins), dtype=np.float64)
    densities = []
    for _, t in tqdm(list(enumerate(times))):
        U = U_of_t(float(t))
        A = truncated_propagator_A(U, N, i=i, j=j)
        _, theta = singular_values_and_thetas(A)

        hist, _ = np.histogram(
            theta,
            bins=theta_edges,
            density=True,
        )
        # densities[ti, :] = hist
        densities.append(hist)

    densities = np.array(densities)
    fig = go.Figure(
        data=go.Surface(
            x=times,
            y=theta_centers,
            z=densities.T,
            colorscale="Viridis",
        )
    )
    fig.update_layout(
        title=f"p̃_(i={i},j={j})(θ,t) (surface view)",
        width=700,
        height=500,
        scene=dict(
            xaxis_title="time t",
            yaxis_title="θ",
            zaxis_title="density",
        ),
    )
    fig.show()
