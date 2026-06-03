"""
Vorzeichen Laplace-Operator (diskret)
Ziel:
    Pruefen, ob der im Code verwendete Stencil ein positiver diskreter
    Laplace-Operator ist. Falls ja, approximiert er -Delta und

        I + tau * L_code

    entspricht in kontinuierlicher Schreibweise

        I - tau * Delta.

Ja, wenn "Laplace" dort der kontinuierliche Operator Delta ist. Im Code ist
L_code die positive Diskretisierung von -Delta.
"""


from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
sys.path.insert(0, str(SOURCE_DIR))

import covariance_kernels as ck 


def symbol_5pt(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """
    Fourier-Symbol des positiven 5-Punkt-Laplace-Stencils.

    Fuer kleine Frequenzen gilt: ell_5(kx, ky) approx kx^2 + ky^2

    Das ist positiv und entspricht damit -Delta, nicht Delta.
    """
    return 4.0 - 2.0 * np.cos(kx) - 2.0 * np.cos(ky)


def symbol_9pt(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """
    Fourier-Symbol des positiven 9-Punkt-Laplace-Stencils.

    Der Stencil ist isotroper als der 5-Punkt-Stencil. Fuer kleine Frequenzen
    gilt hie rauch: ell_9(kx, ky) approx kx^2 + ky^2
    """
    return (
        20.0
        - 8.0 * np.cos(kx)
        - 8.0 * np.cos(ky)
        - 4.0 * np.cos(kx) * np.cos(ky)
    ) / 6.0


def print_local_stencil(name: str, matrix, n: int) -> None:
    """
    Eine Matrixzeile fuer eine innere Zelle als lokales 3x3-Stencil ausgeben.

    Das ist der direkteste Check, welche Nachbarn gekoppelt werden.
    """
    center = (n // 2) * n + (n // 2)
    local = matrix[center].toarray().reshape(n, n)
    r = n // 2
    c = n // 2

    print(f"\n{name}: lokaler Stencil fuer die mittlere Zelle")
    print(local[r - 1 : r + 2, c - 1 : c + 2])


def print_matrix_diagnostics(name: str, matrix, n: int) -> None:
    """
    Numerische Checks fuer den endlichen Gitteroperator.

    Hinweis:
        Wegen der Randbehandlung ist das endliche Matrixspektrum nicht exakt
        dasselbe wie das periodische Fourier-Symbol. Die Vorzeicheninformation
        ist trotzdem sichtbar: die Eigenwerte sind positiv.
    """
    dense = matrix.toarray()
    eig = np.linalg.eigvalsh(dense)
    ones = np.ones(n * n)
    row_sums = dense @ ones

    print(f"\n{name}: Matrixdiagnose")
    print(f"  shape:        {matrix.shape}")
    print(f"  nnz:          {matrix.nnz}")
    print(f"  eig min/max:  {eig[0]:.8g} / {eig[-1]:.8g}")
    print(f"  all eig > 0:  {np.all(eig > 0)}")
    print(f"  ||L ones||:   {np.linalg.norm(row_sums):.8g}")
    print(f"  row sum min/max: {row_sums.min():.8g} / {row_sums.max():.8g}")

    print(
        "  Interpretation: positive Eigenwerte bedeuten, dass dieser diskrete "
        "Operator die positive Variante von -Delta ist."
    )


def print_symbol_diagnostics() -> None:
    """
    Fourier-Symbole fuer kleine Frequenzen ausgeben.

    Wenn ell(kx, ky) ungefaehr kx^2 + ky^2 ist, dann ist der Stencil eine
    positive Diskretisierung von -Delta.
    """
    points = [
        (0.05, 0.00),
        (0.05, 0.05),
        (0.10, 0.00),
        (0.10, 0.10),
    ]

    print("\nFourier-Symbol-Check fuer kleine Frequenzen")
    print("kx      ky      k^2        ell_5      ell_9")
    for kx, ky in points:
        k2 = kx * kx + ky * ky
        ell5 = float(symbol_5pt(kx, ky))
        ell9 = float(symbol_9pt(kx, ky))
        print(f"{kx:0.3f}   {ky:0.3f}   {k2:0.6f}   {ell5:0.6f}   {ell9:0.6f}")

    print(
        "Wenn ell_5 und ell_9 nahe bei k^2 liegen, ist das die Konvention L_code approx -Delta."
    )


def main() -> None:
    n = 9

    # Mit tau=0 und alpha=1 bekommen wir nur den nackten Stencil-Operator L.
    # Die Funktionen heissen precision_matern*, geben in diesem Spezialfall
    # aber den Laplace-Anteil zurueck.
    L5 = ck.precision_matern(n, tau=0.0, alpha=1.0)
    L9 = ck.precision_matern_9pt(n, tau=0.0, alpha=1.0)

    print("Vorzeichencheck fuer den Laplace-Operator")
    # Kontinuierlich gilt: Delta exp(i k x) = -|k|^2 exp(i k x)."
    # also siEin positiver diskreter Operator mit Symbol +|k|^2 approximiert also -Delta."
    

    print_local_stencil("5pt", L5, n)
    print_local_stencil("9pt", L9, n)

    print_matrix_diagnostics("5pt", L5, n)
    print_matrix_diagnostics("9pt", L9, n)

    print_symbol_diagnostics()


if __name__ == "__main__":
    main()

"""
Das Vorzeichen '+ tau * L_code' sollte konsistent mit der kontinuierlichen
Matern-Schreibweise '(I - tau * Delta)^-1', weil L_code die positive Diskretisierung von -Delta ist.

https://nu-cem.github.io/Computational_Physics/notebooks/finite_difference.html

https://notebook.community/eramirem/numerical-methods-pdes/05_elliptic

https://scicomp.stackexchange.com/questions/37656/tensor-product-representation-for-the-9-point-finite-difference-approximations-f
"""
