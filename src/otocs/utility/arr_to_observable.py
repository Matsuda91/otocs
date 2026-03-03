import itertools

import numpy as np
from qulacs import Observable

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I, "X": X, "Y": Y, "Z": Z}


def _kron_all(mats: list[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def matrix_to_observable(H: np.ndarray, n: int, tol: float = 1e-12) -> Observable:
    H = np.asarray(H, dtype=complex)
    dim = 2**n
    if H.shape != (dim, dim):
        raise ValueError(f"shape must be {(dim, dim)}, got {H.shape}")

    if not np.allclose(H, H.conj().T, atol=1e-10):
        raise ValueError("H must be Hermitian (within tolerance)")

    obs = Observable(n)

    for labels in itertools.product(["I", "X", "Y", "Z"], repeat=n):
        P = _kron_all([PAULI[l] for l in labels])
        coef = np.trace(P @ H) / (2**n)

        if abs(coef) > tol:
            terms = []
            for q, l in enumerate(labels):
                if l != "I":
                    terms.append(f"{l} {q}")
            pauli_string = " ".join(terms) if terms else "I 0"
            obs.add_operator(coef, pauli_string)

    return obs
