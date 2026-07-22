import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


datafile = "italy_ingv_rotated_rect_events_declustered_Mc_2.5_eta_-4.60.csv"
outfile = "italy_ingv_rotated_rect_events_declustered_Mc_2.5_eta_-4.60_binned.csv"

dkm = 20
d = pd.read_csv(datafile, sep="|")
I = d["kept"]
dd = d[I].copy()

nbinx = int(np.ceil((dd["x_proj_km"].max() - dd["x_proj_km"].min()) / dkm))
nbiny = int(np.ceil((dd["y_proj_km"].max() - dd["y_proj_km"].min()) / dkm))

i = np.digitize(dd["x_proj_km"], dd["x_proj_km"].min() + np.arange(nbinx + 1) * dkm) - 1
j = np.digitize(dd["y_proj_km"], dd["y_proj_km"].min() + np.arange(nbiny + 1) * dkm) - 1

res = []

for ii in range(nbinx):
    for jj in range(nbiny):
        I = (i == ii) & (j == jj)
        res.append(
                {
                    "bin_ix": ii,
                    "bin_iy": jj,
                    "n_events": np.sum(I),
                    "x_center_km": dd["x_proj_km"].min() + (ii + 0.5) * dkm,
                    "y_center_km": dd["y_proj_km"].min() + (jj + 0.5) * dkm,
                }
        )

pd.DataFrame(res).to_csv(outfile, index=False, sep="|")


resd = np.array([r["n_events"] for r in res]).reshape((nbinx, nbiny)).T


plt.imshow(resd, origin="lower", cmap="viridis", extent=[0, nbinx * dkm, 0, nbiny * dkm])

plt.show()
