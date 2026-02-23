from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt
from pyqsp import angle_sequence, response
from pyqsp.poly import PolyTaylorSeries

TargetFunction = Callable[[npt.NDArray], npt.NDArray]


class QSPPhiSet:
    """
    Quantum Signal Processing (QSP) phase sequence generator.

    This class computes QSP phase angles that implement a target polynomial
    function using quantum signal processing techniques. It uses Chebyshev
    interpolation and symmetric QSP to find the phase sequence.

    Attributes:
        target_func (TargetFunction): Target function to approximate. Should accept
            and return numpy arrays. If None, defaults to lambda x: np.cos(3*x).
        polydeg (int): Degree of the polynomial approximation. If None, defaults to 10.
        max_scale (float): Maximum norm (<1) for rescaling the polynomial.
            If None, defaults to 0.9.
        phiset (np.ndarray): Generated QSP phase angles. Available after calling generate().

    Example:
        >>> phi_set = QSPPhiSet()
        >>> phi_set.generate()
        >>> phi_set.plot()
    """

    def __init__(
        self,
        target_func: TargetFunction | None = None,
        polydeg: int | None = None,
        max_scale: float | None = None,
    ):
        self.target_func = target_func
        self.polydeg = polydeg
        self.max_scale = max_scale
        self.phiset = None

    @property
    def true_func(self):
        return lambda x: self.max_scale * self.target_func(x)

    def generate(self, return_phiset: bool = False):
        # Specify definite-parity target function for QSP.

        if self.target_func is None:
            self.target_func = lambda x: np.cos(3 * x)

        if self.polydeg is None:
            self.polydeg = 10  # Desired QSP protocol length.
        if self.max_scale is None:
            self.max_scale = 0.9  # Maximum norm (<1) for rescaling.

        """
        With PolyTaylorSeries class, compute Chebyshev interpolant to degree
        'polydeg' (using twice as many Chebyshev nodes to prevent aliasing).
        """
        poly = PolyTaylorSeries().taylor_series(
            func=self.target_func,
            degree=self.polydeg,
            max_scale=self.max_scale,
            chebyshev_basis=True,
            cheb_samples=2 * self.polydeg,
        )

        # Compute full phases (and reduced phases, parity) using symmetric QSP.
        (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
            poly, signal_operator="Wz", method="sym_qsp", chebyshev_basis=True
        )

        self.phiset = phiset

        if return_phiset:
            return {"phiset": phiset, "red_phiset": red_phiset, "parity": parity}

    def plot(self):
        """
        Plot response according to full phases.
        Note that `pcoefs` are coefficients of the approximating polynomial,
        while `target` is the true function (rescaled) being approximated.
        """
        response.PlotQSPResponse(
            self.phiset,
            pcoefs=self.polydeg,
            target=self.true_func,
            sym_qsp=True,
            simul_error_plot=True,
        )
