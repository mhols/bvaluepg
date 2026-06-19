"""Prueft die tatsaechliche Korrelationslaenge des 5-Punkt-Priors.

Ziel des Experiments
--------------------
Im bisherigen Modell wird aus dem Eingabeparameter 'rho' zuerst 'alpha(rho)'
berechnet. Daraus entstehen der Basisoperator A und die Prior-Precision P0:

    L  = positiver diskreter 5-Punkt-Laplacian
    A  = I + alpha(rho) * L
    P0 = A.T @ A

Ich will  mal sehen, ob das eingegebene 'rho' nach der Quadrierung des
Operators weiterhin der raeumlichen Korrelationslaenge des fertigen
Gaussian-Priors entspricht. 
(Das sollte zumindest in der Mitte des Gitters der Fall sein, wo die Randbedingungen keinen Einfluss haben oder?
Ansonsten ist rho mehr der besprochene Parameter.)

"""

from dataclasses import dataclass
from pathlib import Path
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as sla


SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE_DIR))

from covariance_kernels import laplacian_2d


DIRECTIONS = {
    "x": ((0, -1), (0, 1)),
    "y": ((-1, 0), (1, 0)),
    "diagonal": ((-1, -1), (-1, 1), (1, -1), (1, 1)),
}


def alpha_from_rho(rho):
    """Achsenkalibrierung des unquadrierten Operators A."""
    if rho <= 0:
        raise ValueError("rho must be positive")
    return 0.5 / (np.cosh(1.0 / rho) - 1.0)


def first_efolding_crossing(distances, correlations):
    """Interpoliert den ersten Schnitt der Korrelation mit 1/e."""
    target = np.exp(-1.0)
    for i in range(1, len(correlations)):
        if correlations[i - 1] > target >= correlations[i]:
            weight = (target - correlations[i - 1]) / (
                correlations[i] - correlations[i - 1]
            )
            return distances[i - 1] + weight * (
                distances[i] - distances[i - 1]
            )
    return np.nan


@dataclass
class Result:
    boundary: str
    rho: float
    direction: str
    distance: float
    profile_distances: np.ndarray
    correlations: np.ndarray


class PrecisionCorrelationExperiment:
    def __init__(
        self,
        ny=41,
        nx=41,
        rhos=(0.5, 1.0, 2.0, 4.0, 8.0),
        boundaries=("zero", "symmetric"),
    ):
        self.ny = ny
        self.nx = nx
        self.rhos = tuple(rhos)
        self.boundaries = tuple(boundaries)
        self.cy = ny // 2
        self.cx = nx // 2
        self.center = self.cy * nx + self.cx

    def build_operators(self, rho, boundary):
        """Baut L, A und die unskalierte Precision P0=A.T@A."""
        L = laplacian_2d(
            self.ny, self.nx, boundary=boundary
        ).tocsc()
        alpha = alpha_from_rho(rho)
        A = sps.eye(self.ny * self.nx, format="csc") + alpha * L
        P0 = (A.T @ A).tocsc()
        return L, A, P0

    def _ray_coordinates(self, rays):
        """Koordinaten gleich langer Strahlen ab der mittleren Zelle."""
        lengths = []
        for dy, dx in rays:
            length = 0
            while (
                0 <= self.cy + (length + 1) * dy < self.ny
                and 0 <= self.cx + (length + 1) * dx < self.nx
            ):
                length += 1
            lengths.append(length)

        length = min(lengths)
        return [
            [
                (self.cy + step * dy, self.cx + step * dx)
                for step in range(length + 1)
            ]
            for dy, dx in rays
        ]

    def correlation_profile(self, factor, covariance_center, rays):
        """Mittelt die Pearson-Korrelation ueber symmetrische Strahlen.

        Fuer eine Zelle i gilt

            corr(i, center)
              = Sigma[i, center] / sqrt(Sigma[i, i] Sigma[center, center]).

        Die benoetigten Diagonaleintraege Sigma[i, i] werden mit derselben
        Sparse-LU-Faktorisierung berechnet. Eine dichte Inverse wird vermieden.
        """
        ray_coordinates = self._ray_coordinates(rays)
        center_variance = covariance_center[self.center]
        variance_cache = {self.center: center_variance}

        correlations = []
        for step in range(len(ray_coordinates[0])):
            values = []
            for ray in ray_coordinates:
                y, x = ray[step]
                index = y * self.nx + x

                if index not in variance_cache:
                    delta = np.zeros(self.ny * self.nx)
                    delta[index] = 1.0
                    variance_cache[index] = factor.solve(delta)[index]

                values.append(
                    covariance_center[index]
                    / np.sqrt(
                        center_variance * variance_cache[index]
                    )
                )
            correlations.append(np.mean(values))

        step_length = (
            np.sqrt(2.0) if len(rays) == 4 else 1.0
        )
        distances = step_length * np.arange(len(correlations))
        return distances, np.asarray(correlations)

    def analyze(self):
        results = []
        dimension = self.ny * self.nx
        delta_center = np.zeros(dimension)
        delta_center[self.center] = 1.0

        for boundary in self.boundaries:
            for rho in self.rhos:
                _, A, P0 = self.build_operators(rho, boundary)
                factor = sla.splu(P0)
                covariance_center = factor.solve(delta_center)

                # A and P0 muessen strikt positiv definit sein.
                min_eigenvalue_A = sla.eigsh(
                    A, k=1, which="SA", return_eigenvectors=False
                )[0]
                min_eigenvalue_P0 = sla.eigsh(
                    P0, k=1, which="SA", return_eigenvectors=False
                )[0]

                print(
                    f"\nboundary={boundary}, rho={rho:g}, "
                    f"alpha={alpha_from_rho(rho):.6g}, "
                    f"min_eig(A)={min_eigenvalue_A:.6g}, "
                    f"min_eig(P0)={min_eigenvalue_P0:.6g}"
                )

                for direction, rays in DIRECTIONS.items():
                    distances, correlations = self.correlation_profile(
                        factor, covariance_center, rays
                    )
                    length = first_efolding_crossing(
                        distances, correlations
                    )
                    print(
                        f"  {direction:8s}: length={length:.4g}, "
                        f"length/rho={length / rho:.4g}"
                    )
                    results.append(
                        Result(
                            boundary,
                            rho,
                            direction,
                            length,
                            distances,
                            correlations,
                        )
                    )
        return results

    def plot(self, results):
        """Zeigt Korrelationsprofile und gemessene 1/e-Laengen."""
        fig, axes = plt.subplots(
            3, 2, figsize=(11, 11), sharey=True, constrained_layout=True
        )

        for row, direction in enumerate(DIRECTIONS):
            for column, boundary in enumerate(self.boundaries):
                ax = axes[row, column]
                for rho in self.rhos:
                    result = next(
                        item
                        for item in results
                        if item.boundary == boundary
                        and item.rho == rho
                        and item.direction == direction
                    )
                    ax.plot(
                        result.profile_distances,
                        result.correlations,
                        marker="o",
                        markersize=3,
                        label=f"rho={rho:g}",
                    )
                ax.axhline(np.exp(-1.0), color="black", ls="--", label="1/e")
                ax.set(
                    title=f"{direction}, boundary={boundary}",
                    xlabel="distance [grid cells]",
                    ylabel="Pearson correlation",
                    ylim=(-0.05, 1.05),
                )
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)

        fig2, axes2 = plt.subplots(
            1, 2, figsize=(11, 4.5), sharex=True, sharey=True,
            constrained_layout=True,
        )
        for ax, boundary in zip(axes2, self.boundaries):
            for direction in DIRECTIONS:
                selected = [
                    item
                    for item in results
                    if item.boundary == boundary
                    and item.direction == direction
                ]
                ax.plot(
                    [item.rho for item in selected],
                    [item.distance for item in selected],
                    marker="o",
                    label=direction,
                )
            ax.plot(
                self.rhos, self.rhos, "k--", label="effective length = rho"
            )
            ax.set(
                title=f"boundary={boundary}",
                xlabel="input rho",
                ylabel="measured 1/e correlation length",
            )
            ax.grid(alpha=0.25)
            ax.legend()

        plt.show()

    def run(self):
        results = self.analyze()
        self.plot(results)
        return results


if __name__ == "__main__":
    PrecisionCorrelationExperiment().run()

'''
Die Kovarianz des Priors ist Sigma0 = P0^{-1}. Ausgehend von der mittleren
Gitterzelle berechnet das Experiment die Korrelation entlang der
x-Achse, der y-Achse und der Diagonalen. Als effektive Korrelationslaenge wird
die erste Distanz verwendet, bei der die Korrelation auf 1/e gefallen ist.

Die Rechnung wird fuer mehrere Werte von 'rho' sowie fuer die
Randbedingungen 'zero' und 'symmetric' wiederholt. Dadurch wird zugleich
sichtbar, ab welcher raeumlichen Reichweite die Randbedingung das Ergebnis
beeinflusst.

Offensichtlich ist die gemessene Korrelationslaenge nicht exakt gleich dem eingegebenen 'rho',
aber sie ist in einem aehnlichen Bereich. Das Experiment zeigt, dass die Korrelationslaenge des
5-Punkt-Priors in etwa proportional zu 'rho' ist, aber nicht exakt uebereinstimmt. 
'''
