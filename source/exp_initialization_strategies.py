from polyagammadensity import PolyaGammaDensity
import syntheticdata as sd
import numpy as np
import matplotlib.pyplot as plt


# --- setup / basic config ---
n, m = 20, 20
lam = 10

pgd = PolyaGammaDensity(
    prior_mean=np.zeros(n*m),
    prior_covariance=sd.spatial_covariance_gaussian(n, m, 3, 1),
    lam=lam
)

# --- data generation (wahres model / ground truth) ---
f_true = pgd.random_prior_parameters()
events = pgd.random_events_from_f(f_true)
pgd.set_data(events)

# A) inverse crime (quick sanity check)
# f0_prior = pgd.prior_mean.copy()
f0_prior = pgd.prior_mean + 0.1*np.random.normal(size=n*m)

# B) heuristisch from data (aktueller approach)
s = (events - np.sqrt(events)) / lam
s = np.clip(s, 0.1, 0.9)
f0_data = np.log(s / (1 - s))

# C) filtered data start (low-pass before logit)
from scipy.ndimage import gaussian_filter
events_img = sd.scanorder_to_image(events, n, m)
events_smooth = gaussian_filter(events_img, sigma=1.5)
events_smooth = sd.image_to_scanorder(events_smooth)

s = events_smooth / lam
s = np.clip(s, 0.1, 0.9)
f0_filtered = np.log(s / (1 - s))  # sigmoid(f0_filtered) ~= s (by construction)

# let's try the different start guesses and see how MAP behaves
for name, f0 in {
    "prior": f0_prior,
    "data": f0_data,
    "filtered": f0_filtered
}.items():

    f_map = pgd.max_logposterior_estimator(f0, niter=100)

    # compute gradient at MAP (sollte nahe 0 sein)
    grad_map = pgd.neg_grad_logposterior(f_map)
    plt.figure()
    plt.title(f"Gradient at MAP estimate ({name})")
    plt.imshow(sd.scanorder_to_image(np.abs(grad_map), n, m).T)
    plt.colorbar()
    
    plt.figure()
    plt.title(f"starting estimate ({name})")
    plt.imshow(sd.scanorder_to_image(pgd.field_from_f(f0), n, m).T)
    plt.colorbar()


    plt.figure()
    plt.title(f"MAP estimate ({name})")
    plt.imshow(sd.scanorder_to_image(pgd.field_from_f(f_map), n, m).T)
    plt.colorbar()

# schauen wir uns mal die wahren values an
field_true = pgd.field_from_f(f_true)

plt.figure()
plt.title("True parameter f_true")
plt.imshow(sd.scanorder_to_image(f_true, n, m).T)
plt.colorbar()

plt.figure()
plt.title("True field (lam * sigmoid(f_true))")
plt.imshow(sd.scanorder_to_image(field_true, n, m).T)
plt.colorbar()

plt.figure()
plt.title("Observed events (counts)")
plt.imshow(sd.scanorder_to_image(events, n, m).T)
plt.colorbar()

plt.show()

print("Done")



'''
links:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html#scipy.ndimage.gaussian_filter
https://numpy.org/devdocs/reference/generated/numpy.clip.html
'''