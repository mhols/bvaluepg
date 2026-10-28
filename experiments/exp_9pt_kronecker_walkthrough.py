"""9-Punkt-Stencil schrittweise auf einem kleinen rechteckigen Gitter pruefen.

erkenntnisse:
"""

from pathlib import Path
import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

# specify the source directory for polyagammapoisson package
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
spec = importlib.util.spec_from_file_location(
    "polyagammapoisson", SOURCE_DIR / "__init__.py",
    submodule_search_locations=[str(SOURCE_DIR)],
)

package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from polyagammapoisson import covariance_kernels as ck


# specify grid size, boundary condition, and whether to show plots
NY, NX = 4, 5
BOUNDARY = "zero"  # auch "symmetric" ausprobieren

def direct_stencil(field, boundary):
    """Unabhaengige Referenz: Ghost-Werte null oder Randzelle wiederholen."""
    ny, nx = field.shape
    weights = np.array([[-1, -4, -1], [-4, 20, -4], [-1, -4, -1]]) / 6
    result = np.zeros_like(field, dtype=float)
    for iy in range(ny):
        for ix in range(nx):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    jy, jx = iy + dy, ix + dx
                    if boundary == "zero":
                        if not (0 <= jy < ny and 0 <= jx < nx):
                            continue
                    elif boundary == "symmetric":
                        jy = min(max(jy, 0), ny - 1)
                        jx = min(max(jx, 0), nx - 1)
                    else:
                        raise ValueError(boundary)
                    result[iy, ix] += weights[dy + 1, dx + 1] * field[jy, jx]
    return result


# 1. Beschriftetes Feld und C-order: x laeuft im Vektor am schnellsten.
field = np.arange(1, NY * NX + 1, dtype=float).reshape(NY, NX)
vector = field.ravel(order="C")
print("Feld U (Zeile y, Spalte x):\n", field)
print("C-order-Vektor:\n", vector)

# 2. Bx/By summieren die beiden Nachbarn in der jeweiligen Richtung.
Ix, Iy = sps.eye(NX, format="csr"), sps.eye(NY, format="csr")
Lx = ck.laplacian_1d(NX, boundary=BOUNDARY)
Ly = ck.laplacian_1d(NY, boundary=BOUNDARY)
Bx, By = 2 * Ix - Lx, 2 * Iy - Ly
horizontal = sps.kron(Iy, Bx, format="csr")
vertical = sps.kron(By, Ix, format="csr")
diagonal = sps.kron(By, Bx, format="csr")
print(f"\nRandregel: {BOUNDARY}")
print("Bx:\n", Bx.toarray())
print("By:\n", By.toarray())
for name, operator in (("links + rechts", horizontal),
                       ("oben + unten", vertical),
                       ("vier Diagonalen", diagonal)):
    print(f"\n{name}:\n", (operator @ vector).reshape(NY, NX))

# 3. Lokales Stencil: (20 U - 4 * direkte Nachbarn - Diagonalen) / 6
L9_parts = (20 * sps.eye(NY * NX) - 4 * (horizontal + vertical) - diagonal) / 6
L9 = ck.laplacian_9pt_2d(NY, NX, boundary=BOUNDARY)
np.testing.assert_allclose(L9.toarray(), L9_parts.toarray(), atol=1e-12)
result = (L9 @ vector).reshape(NY, NX)
np.testing.assert_allclose(result, direct_stencil(field, BOUNDARY), atol=1e-12)
print("\nL9 @ U:\n", result)

# Eine Matrixzeile zeigt, welche Eingabezellen eine bestimmte Ausgabe nutzt.
positions = ((1, 2), (0, 2), (0, 0))  # Innenpunkt, Kante, Ecke
for iy, ix in positions:
    index = iy * NX + ix
    print(f"\nStencil der Ausgabe ({iy}, {ix}), Vektorindex {index}:\n",
          L9.getrow(index).toarray().reshape(NY, NX))

# 4. Vollstaendiger Operatorvergleich gegen direkte Offsets, beide Randregeln.
# Rechtecke und Singleton-Achsen; alle Basisvektoren statt nur ein Testbild.
for shape in ((3, 4), (4, 3), (1, 4), (4, 1), (1, 1)):
    size = int(np.prod(shape))
    for boundary in ("zero", "symmetric"):
        matrix = ck.laplacian_9pt_2d(*shape, boundary=boundary).toarray()
        expected = np.column_stack([
            direct_stencil(np.eye(size)[:, j].reshape(shape), boundary).ravel()
            for j in range(size)
        ])
        np.testing.assert_allclose(matrix, expected, atol=1e-12)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-12
        if boundary == "symmetric":
            np.testing.assert_allclose(matrix @ np.ones(size), 0, atol=1e-12)
print("Das prueft den Laplacian, noch keine Posterior-Kalibrierung oder Konvergenz.")


panels = [(field, "U"),
            ((horizontal @ vector).reshape(NY, NX), "Links + rechts"),
            ((vertical @ vector).reshape(NY, NX), "Oben + unten"),
            ((diagonal @ vector).reshape(NY, NX), "Diagonalen"),
            (result, "L9 U")]
fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.5), constrained_layout=True)
for ax, (values, title) in zip(axes, panels):
    im = ax.imshow(values, origin="upper", cmap="coolwarm")
    ax.set(title=title, xlabel="x-Spalte", ylabel="y-Zeile")
    for iy in range(NY):
        for ix in range(NX):
            ax.text(ix, iy, f"{values[iy, ix]:g}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.65)
fig.suptitle(f"9-Punkt-Kronecker-Walkthrough, boundary={BOUNDARY}")
plt.show()
