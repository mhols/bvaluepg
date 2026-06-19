import matplotlib.pyplot as plt
import numpy as np

import covariance_kernels as ck
import polyagammadensity as pgd

ny, nx = 20, 35

precision = ck.precision_matern(
    ny,
    nx,
    rho=3,
    v2=1,
    boundary="symmetric",
    )

model = pgd.PolyaGammaDensity2D(
    prior_mean=np.zeros(ny * nx),
    prior_precision=precision,
    sparse=True,
    lam=10,
    n=ny,
    m=nx,
)

f = model.random_prior_parameters()
rate = model.field_from_f(f)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].imshow(f.reshape(ny, nx), origin="lower")
axes[0].set_title("Latentes Prior-Sample f")

axes[1].imshow(rate.reshape(ny, nx), origin="lower")
axes[1].set_title("Poisson-Rate")

plt.show()
